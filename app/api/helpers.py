import asyncio
import json
import dill as dl
from datetime import datetime
from greattunes import TuneSession
from greattunes.data_format_mappings import tensor2pretty_covariate
from app.db import Experiment, User, PublicExperiment, PublicExperimentAsk


# custom json encoder for sets
class SetEncoder(json.JSONEncoder):
    def default(self, obj):
       if isinstance(obj, set):
          return list(obj)
       return json.JSONEncoder.default(self, obj)


# helper class for storing and reading
class ParseModel:

    # create method to parse covars to the original object required for TuneSession
    @staticmethod
    def _dict_replace_type_value(d):
        '''
        in the covars dict coming from experiment/new endpoint, convert the values for 'type' keyword to data types
        (str, int, float) as required by TuneSession
        :param d (dict of covars):
        :return x (dict of covars with values for keyword "type" updated):
        '''

        x = {}
        for k, v in d.items():
            if isinstance(v, dict):
                v = ParseModel._dict_replace_type_value(v)
            elif k == "type":
                if v == "str":
                    v = str
                elif v == "float":
                    v = float
                elif v == "int":
                    v = int
            x[k] = v
        return x

    @staticmethod
    def parse_covars_dict(covars):
        '''
        parse covars in dict form (from posting to experiment/new endpoint). Updates the type from str to correct type
        :param covars (dict)
        :return: covars_out (dict)
        '''

        covars_out = ParseModel._dict_replace_type_value(covars)

        return covars_out

    @staticmethod
    def dump_iteration_numbers(cls):
        '''
        get iteration numbers from instantiated class of type TuneSession
        :param cls (TuneSession object): instantiated TuneSession object
        :return covars_sampled_iter (int)
        :return response_sampled_iter (int)
        '''

        covars_sampled_iter = cls.model["covars_sampled_iter"]
        response_sampled_iter = cls.model["response_sampled_iter"]

        return covars_sampled_iter, response_sampled_iter


    @staticmethod
    def dump_best_response_to_json(cls):
        '''
        get best response and the covariates corresponding to best response, and dump them to JSON. At initialization of
        experiment (before first step to find optimum has happened), the values will be None
        :param cls (TuneSession object): the values of interest are kept in the attributes best_response and
        covars_best_response (both are pandas df or None)
        :return best_response_json (JSON)
        :return covars_best_response_json (JSON)
        '''

        best_response_json = None
        if cls.best_response is not None:
            best_response_json = cls.best_response.reset_index(drop=True).to_json()

        covars_best_response_json = None
        if cls.covars_best_response is not None:
            covars_best_response_json = cls.covars_best_response.reset_index(drop=True).to_json()

        return best_response_json, covars_best_response_json

    @staticmethod
    def dump_model_object_binary_to_string(mdl):
        '''
        dump instantiated TuneSession model object to str format, e.g. for storage in db
        :param mdl (TuneSession object)
        :return: mdl_str (str)
        '''

        return dl.dumps(mdl)

    @staticmethod
    def load_model_object_binary_from_string(mdl_str):
        '''
        load model object serialized to str via ParseModel.dump_model_object_binary_to_string into model object again
        :param mdl_str (output from ParseModel.dump_model_object_binary_to_string)
        :return: mdl (instatiated TuneSession object)
        '''

        return dl.loads(mdl_str)


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


