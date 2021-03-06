import json
import dill as dl


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
