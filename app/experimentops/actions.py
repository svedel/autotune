import json
from datetime import datetime

import pandas as pd
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from greattunes import TuneSession
from greattunes.data_format_mappings import tensor2pretty_covariate

from app.db import Experiment, User, PublicExperiment, PublicExperimentAsk, PublicExperimentTell
from app.experimentops.utils import ParseModel, SetEncoder


# class for operations on experiments
class ExperimentOperations:

    @staticmethod
    def create_experiment_model_object(covars, model, acq_func, **kwargs):
        cls = TuneSession(covars=covars, model=model, acq_func=acq_func)
        return cls

    @staticmethod
    def parse_new_experiment(new_exp, user):
        '''
        parse input provided to /experiment/new endpoint
        :param new_exp (object of type PublicCreateExperiment):
        :param user (object of type User): user to which this experiment will be assigned
        :return exp (object of type Experiment): can be saved to db via exp.save() operation
        :return mdl (initialized TuneSession model object)
        '''

        # get dict
        new_exp_dict = new_exp.dict()

        # parse covars via method in ParseModel
        covars_parsed = ParseModel.parse_covars_dict(covars=new_exp_dict["covars"])

        # initialize TuneSession
        mdl = ExperimentOperations.create_experiment_model_object(covars=covars_parsed,
                                                                  model=new_exp_dict["model_type"].value,
                                                                  acq_func=new_exp_dict["acq_func"].value)

        # number of iterations taken in tuning
        covars_sampled_iter, response_sampled_iter = ParseModel.dump_iteration_numbers(mdl)

        # get best response (best result for optimum so far) and corresponding covariate values. This method applies at
        # any iteration number, but the result at initialization will be None
        best_response_json, covars_best_response_json = ParseModel.dump_best_response_to_json(mdl)

        # create entry for db
        exp = Experiment(name=new_exp_dict["name"],
                         description=new_exp_dict["description"],
                         covars=json.dumps(new_exp_dict["covars"], cls=SetEncoder),
                         model_type=new_exp_dict["model_type"].value,
                         acq_func_type=new_exp_dict["acq_func"].value,
                         covars_sampled_iter=covars_sampled_iter,
                         response_sampled_iter=response_sampled_iter,
                         best_response=best_response_json,
                         covars_best_response=covars_best_response_json,
                         user=user,  # user is foreign key to user table
                         model_object_binary=ParseModel.dump_model_object_binary_to_string(mdl),
                         )

        return exp


    @staticmethod
    async def public_experiment(exp_uuid):
        '''
        extract entry in 'experiments' table based on exp_uuid, and returns in format PublicExperiment
        :param exp_uuid:
        :return:
        '''

        # identify the right experiment
        exp = await Experiment.objects.get(Experiment.exp_uuid == exp_uuid)

        # identify the associated user (a foreign key relation) via the primary key
        user = await User.objects.get(User.id == exp.user.pk)

        public_exp = PublicExperiment(
            exp_uuid=exp.exp_uuid,
            name=exp.name,
            description=exp.description,
            time_created=exp.time_created,
            time_updated=exp.time_updated,
            covars=exp.covars,
            model_type=exp.model_type,
            acq_func_type=exp.acq_func_type,
            active=exp.active,
            best_response=exp.best_response,
            covars_best_response=exp.covars_best_response,
            covars_sampled_iter=exp.covars_sampled_iter,
            response_sampled_iter=exp.response_sampled_iter,
            user_uuid=user.uuid
        )

        return public_exp


    @staticmethod
    async def ask_next_datapoint(exp):
        '''
        retrieves the next datapoint based on TuneSession 'ask'-method. Next datapoint will be determined based off
        previous data points, the acquisition function and the model type
        :param exp (Experiment entry): stored experiment
        :return:
        '''

        # load model
        model_object = ParseModel.load_model_object_binary_from_string(exp.model_object_binary)

        # find next datapoint, will be available as last entry in model_object.proposed_X (in torch double tensor
        # format)
        model_object.ask()

        # display next datapoint as json
        proposed_covars_json = ExperimentOperations._proposed_covars_json_for_API_return(model_object)

        # update model binary in db
        exp.model_object_binary = ParseModel.dump_model_object_binary_to_string(model_object)
        exp.time_updated = datetime.utcnow()
        await exp.update(_columns=["model_object_binary", "time_updated"])  # updates fields in database
        await exp.load()  # loads the latest stored data (in order to get the timestamp)

        # define new exp data model class just for new covariates, cast data into that class and return it to the route
        # to be returned via API
        ask_exp = PublicExperimentAsk(
            exp_uuid=exp.exp_uuid,
            time_updated=exp.time_updated,
            covars_next_exp=proposed_covars_json
        )

        return ask_exp


    @staticmethod
    def _proposed_covars_json_for_API_return(model_object):
        '''
        display output of .ask-method for return via API
        :param model_object: instantiated TuneSession object where .ask method has just been run (covariates for next experiment have been found)
        :return json (json): proposed covariates for next experiment, returned in json format
        '''

        # number of covariates
        num_covars = model_object.proposed_X[-1].size()[0]

        # convert to pandas
        proposed_df = tensor2pretty_covariate(
            train_X_sample=model_object.proposed_X[-1].reshape(1, num_covars),
            covar_details=model_object.covar_details
        )

        return proposed_df.to_json(orient="records")

    @staticmethod
    def _process_json_to_pandas(input_json):
        '''
        assumes input_json is as a pandas df converted to json string using .to_json() method. This method parses this
        string and returns the pandas df, and raises an exception if this is not possible
        :param input_json (json str of pandas df (one row))
        :return input_df (pandas df)
        '''

        try:
            # reads covars to pandas
            #input_df = pd.read_json(input_json)
            input_df = pd.DataFrame(jsonable_encoder(input_json))

            return input_df

        except ValueError:
            raise HTTPException(status_code=422, detail="Unprocessable entity: input json " + str(input_json))

    @staticmethod
    def _verify_df_content_type(df, covar_details, content_type="covars"):
        '''
        verifies column data type of each field in pandas dataframe df, assuming it is of type 'covars' or
        'response'. For 'covars' will use covar_details as comparison.

        Column names in df must match names in model_object.covar_details for covariates. For response, there must be
        only a single column of name 'Response'

        Any missing columns (as judged by column names) will be ignored

        :param df (pandas dataframe):
        :param covar_details (covar_details attribute from instantiated object of type TuneSession):
        :param content_type (str; can take values "covars" or "response"):
        :return:
        '''

        # initialize
        verified = False

        # mapping table (pandas uses different types than native python). Recast all to string type and compare strings.
        # Format: {pandas, python}
        mapping_table = {"object": str(str), "float64": str(float), "int64": str(int)}

        # get column names and data types as list of tuples
        if content_type == "response":
            col_types = {"Responses": str(float)}
        elif content_type == "covars":
            col_types = {k: str(v["type"]) for k, v in covar_details.items()}

        # loop through columns
        # for each column, grab the data type and convert to correct str using mapping table
        # check that type is identical to what's in covars_details (str of that), using str compare
        for ct in df.columns.to_list():

            # verify column name in the experiment by checking it exists in covar_details
            if ct in list(col_types.keys()):

                # verify data type
                if not mapping_table[str(df[ct].dtype)] == col_types[ct]:
                    raise TypeError("ExperimentOperations._verify_df_content_type: Expected data type " + col_types[ct]
                                    + " but received " + mapping_table[str(df[ct].dtype)] + " for variable '" + ct + "'")

        verified = True
        return verified

    @staticmethod
    def _verify_df_columns(df, covar_details, content_type="covars"):
        '''
        verifies name of each field in pandas dataframe df, assuming it is of type 'covars' or 'response'.
        For 'covars' will use covar_details as comparison.

        Column names in df must match names in model_object.covar_details for covariates. For response, there must be
        only a single column of name 'Response'

        :param df (pandas dataframe):
        :param covar_details (covar_details attribute from instantiated object of type TuneSession):
        :param content_type (str; can take values "covars" or "response"):
        :return:
        '''

        verified = False

        # reference: columns to verify
        if content_type == "response":
            col_names = ["Response"]
        elif content_type == "covars":
            col_names = list(covar_details.keys())

        # verify that all required column names are there
        df_col_names = df.columns.to_list()
        for cn in col_names:
            if cn not in df_col_names:
                verified = False
                raise NameError("ExperimentOperations._verify_df_columns: Missing expected column " + cn + ", perhaps others.")

        verified = True
        return verified

    @staticmethod
    async def tell_datapoint(exp, covars_tell, response_tell):
        '''
        adds user-provided data for latest experiment to model and updates model in response to this data. Specifically
        does the following
        - verifies data content of user-provided data from latest experiment for covariates and response (verifies all
        column names present and their data types), otherwise raises HTTPExceptions
        - adds data to model (if passes tests), updates (re-trains) model
        - updates 'exp' entry in 'experiment' db table

        TODO: output data in json format is currently coverted to str because of issues handling native json format

        :param exp (object): instantiated object of type Experiment corresponding to an entry in 'experiment' db table
        :param covars_tell (json): pandas df for covariates serialized to json via .to_json method and provided via
        'experiment/tell/{uuid}' endpoint. Each covariate must have its own column
        :param response_tell (json): pandas df for response serialized to json via .to_json method. Must only contain
        on column named "Response" which must be of type float
        :return tell_exp (instantiated data model object of type PublicExperimentTell)
        '''

        # load model and retrieve covar_details
        model_object = ParseModel.load_model_object_binary_from_string(exp.model_object_binary)
        covar_details = model_object.covar_details

        # convert and check content of covars_tell
        covars_df = ExperimentOperations._process_json_to_pandas(input_json=covars_tell)
        try:
            if not ExperimentOperations._verify_df_columns(df=covars_df, covar_details=covar_details,
                                                           content_type="covars"):
                raise ValueError("ExperimentOperations.tell_datapoint: Field names for covariates not accepted.")
            if not ExperimentOperations._verify_df_content_type(df=covars_df, covar_details=covar_details,
                                                                content_type="covars"):
                raise TypeError("ExperimentOperations.tell_datapoint: Reported type for covariates not accepted.")
        except (ValueError, NameError, TypeError):
            raise HTTPException(status_code=422, detail="Unprocessable entity: covariates")

        # convert and check content of response_tell
        response_df = ExperimentOperations._process_json_to_pandas(input_json=response_tell)
        try:

            if not ExperimentOperations._verify_df_columns(df=response_df, covar_details=covar_details,
                                                           content_type="response"):
                raise ValueError("ExperimentOperations.tell_datapoint: Field name for response not accepted.")
            if not ExperimentOperations._verify_df_content_type(df=response_df, covar_details=covar_details,
                                                                content_type="response"):
                raise TypeError("ExperimentOperations.tell_datapoint: Reported type for response not accepted.")
        except (ValueError, NameError, TypeError):
            raise HTTPException(status_code=422, detail="Unprocessable entity: response")

        # report new data to model, save
        model_object.tell(covar_obs=covars_df, response_obs=response_df)

        # get best response
        best_response_json, covars_best_response_json = ParseModel.dump_best_response_to_json(model_object)

        # number of iterations taken in tuning
        covars_sampled_iter, response_sampled_iter = ParseModel.dump_iteration_numbers(model_object)

        # update 'exp' entry in 'experiments' db table
        exp.model_object_binary = ParseModel.dump_model_object_binary_to_string(model_object)
        exp.best_response = best_response_json
        exp.covars_best_response = covars_best_response_json
        exp.covars_sampled_iter = covars_sampled_iter
        exp.response_sampled_iter = response_sampled_iter
        exp.time_updated = datetime.utcnow()

        await exp.update(_columns=["model_object_binary", "best_response", "covars_best_response",
                                   "covars_sampled_iter", "response_sampled_iter", "time_updated"])  # updates fields in database

        # return output to user (new data model)
        await exp.load()

        tell_exp = PublicExperimentTell(
            exp_uuid=exp.exp_uuid,
            covars_tell=str(covars_tell),
            response_tell=str(response_tell),
            best_response=str(exp.best_response),
            covars_best_reponse=str(exp.covars_best_response),
            covars_sampled_iter=exp.covars_sampled_iter,
            response_sampled_iter=exp.response_sampled_iter,
            time_updated=exp.time_updated
        )

        return tell_exp
