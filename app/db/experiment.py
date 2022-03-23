from datetime import datetime
import ormar
import pydantic
from enum import Enum
from pydantic import Json, validator, root_validator, StrictInt, StrictFloat, StrictStr, UUID4
from typing import Set, Dict, Union, Optional
import uuid
from greattunes import TuneSession

from app.db.core import BaseMeta
from app.db.user import User


class VarType(Enum):
    int = "int"
    cont = "cont"
    cat = "cat"


class Variable(pydantic.BaseModel):
    vtype: VarType
    guess: Union[StrictFloat, StrictInt, StrictStr]
    min: Optional[Union[StrictFloat, StrictInt]] = None
    max: Optional[Union[StrictFloat, StrictInt]] = None
    options: Optional[Set[str]] = None

    # this check is needed to make 'type' available for 'check_guess' validator, but it is not otherwise needed since
    # VarType itself ensures type validation
    @validator('vtype', allow_reuse=True)
    def req_check(cls, t):
        assert t.value in ['int', 'cont', 'cat'], "'vtype' must take value from set ['int', 'cont', 'cat']"
        return t

    # add new field called "type"
    @root_validator(pre=False, allow_reuse=True)
    def insert_type(cls, values):
        if values['vtype'].value == 'int':
            values['type'] = 'int'
        elif values['vtype'].value == 'cont':
            values['type'] = 'float'
        elif values['vtype'].value == 'cat':
            values['type'] = 'str'
        return values

    @root_validator(pre=True, allow_reuse=True)
    def set_guessminmax_types(cls, values):
        if values['vtype'] == 'int':
            values['guess'] = int(values['guess'])
            values['min'] = int(values['min'])
            values['max'] = int(values['max'])
        elif values['vtype'] == 'cont':
            values['guess'] = float(values['guess'])
            values['min'] = float(values['min'])
            values['max'] = float(values['max'])
        return values


    # check right data type of 'guess'
    @validator('guess', allow_reuse=True)
    def check_guess_datatype(cls, g, values):
        if values['vtype'].value == 'int':
            assert isinstance(g, int), "data type mismatch between 'guess' and 'vtype'. Expected type 'int' from 'guess' but received " + str(
                type(g))
            return g
        elif values['vtype'].value == 'cont':
            assert isinstance(g, float), "data type mismatch between 'guess' and 'vtype'. Expected type 'float' from 'guess' but received " + str(
                type(g))
            return g
        elif values['vtype'].value == 'cat':
            assert isinstance(g, str), "data type mismatch between 'guess' and 'vtype'. Expected type 'str' from 'guess' but received " + str(
                type(g))
            return g

    # check that 'min' is included for types 'int', 'cont'
    @validator('min', allow_reuse=True)
    def check_min_included(cls, m, values):
        if values['vtype'].value in ['int', 'cont']:
            assert m is not None
            return m

    # check that 'max' is included for types 'int', 'cont'
    @validator('max', allow_reuse=True)
    def check_max_included(cls, m, values):
        if values['vtype'].value in ['int', 'cont']:
            assert m is not None
            return m

    # check that 'options' is included for type 'cat'
    @validator('options', allow_reuse=True)
    def check_options_included(cls, op, values):
        if values['vtype'].value == 'cat':
            assert op is not None
            return op

    # removes all fields which have value None
    @root_validator(pre=False, allow_reuse=True)
    def remove_all_nones(cls, values):
        values = {k: v for k, v in values.items() if v is not None}
        return values

    class Config:
        fields = {"vtype": {"exclude": True}}


class VariableOut(pydantic.BaseModel):
    type: str
    guess: Union[StrictFloat, StrictInt, StrictStr]
    min: Optional[Union[StrictFloat, StrictInt]] = None
    max: Optional[Union[StrictFloat, StrictInt]] = None
    options: Optional[Set] = None

    # removes all fields which have value None
    @root_validator(pre=False, allow_reuse=True)
    def remove_all_nones(cls, values):
        values = {k: v for k, v in values.items() if v is not None}
        return values


# enums of TuneSession model types and acquisition function types
tmp_cls = TuneSession(covars=[(0.5, 0.0, 1.0)])
ModelTypes = Enum('ModelTypes', dict([(x,x) for x in tmp_cls.MODEL_LIST]))
AcqFuncTypes = Enum('AcqFuncTypes', dict([(x,x) for x in tmp_cls.ACQ_FUNC_LIST]))


# public class for creating an experiment via API
class PublicCreateExperiment(pydantic.BaseModel):
    name: str = "Experiment name"
    description: Optional[str] = "A description of the experiment is typically a good idea"
    covars: Dict[str, Variable]  # [variable name, content in form of Variable]
    model_type: Optional[ModelTypes] = "SingleTaskGP"
    acq_func: Optional[AcqFuncTypes] = "ExpectedImprovement"


# db class for creating experiment
class Experiment(ormar.Model):
    '''
    ormar class (data model and database model) for experiments
    '''
    class Meta(BaseMeta):
        tablename: str = "experiments"

    id: int = ormar.Integer(primary_key=True, autoincrement=True)
    exp_uuid: str = ormar.UUID(uuid_format="string", default=uuid.uuid4, index=True)
    name: str = ormar.String(nullable=False, max_length=256)
    description: Optional[str] = ormar.Text(nullable=True)
    time_created: datetime = ormar.DateTime(default=datetime.utcnow, nullable=False)
    time_updated: datetime = ormar.DateTime(default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    active: bool = ormar.Boolean(default=True, nullable=False)
    covars: Json = ormar.JSON(nullable=False)
    model_type: str = ormar.String(max_length=100, choices=list(ModelTypes))
    acq_func_type: str = ormar.String(max_length=100, choices=list(AcqFuncTypes))
    best_response: Json = ormar.JSON(nullable=True)  # best response from model
    covars_best_response: Json = ormar.JSON(nullable=True)  # covariates corresponding to best response from model
    covars_sampled_iter: int = ormar.Integer()
    response_sampled_iter: int = ormar.Integer()
    user: User = ormar.ForeignKey(User, nullable=False)
    model_object_binary: str = ormar.LargeBinary(max_length=1000000, nullable=True) #ormar.Text(nullable=True)  # model file dumped to str


class PublicExperiment(pydantic.BaseModel):
    '''
    data model for returning experiments (data from 'experiment' table)
    '''
    exp_uuid: UUID4
    name: str
    description: Optional[str]
    covars: Dict
    model_type: Optional[str]
    acq_func: Optional[str]
    time_created: datetime
    time_updated: datetime
    active: bool
    best_response: Json
    covars_best_response: Json
    covars_sampled_iter: int
    response_sampled_iter: int
    user_uuid: UUID4
