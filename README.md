![alt text](docs/figs/autotune.png)

Natively async API for Bayesian optimization-based model tuning, with load balancing and a separate production setup.

# Introduction and basic examples 
[Hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) tuning is an important part of the 
machine learning development workflow. Tuning hyperparameters may not seem like a big
challenge for a model with only 1 or 2, but for models where each training session is time consuming or expensive such as 
large models (e.g. deep learning models with hundreds of nodes and tens of hidden layers) or models deployed in 
situations with limited connectivity (e.g. edge analytics), finding good hyperparameters with the fewest iterations is
paramount.

This repo contains a productionized API to achieve this using [Bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization). 
Bayesian optimization is a structured approach which updates the sampling strategy after each set of hyperparameters is 
tested to proposed the next set of hyperparameters which will maximize the expected learning of the model. The workhorse 
is the `PyTorch`-based [`greattunes` Bayesian optimization library](https://pypi.org/project/greattunes/) which allows
for mixed variables of **integers**, **continuous** and **categorical** types.

## Terminology
A tuning session for hyperparameters for a model is called an **"experiment"**. It often contains multiple rounds of
trying hyperparameter candidates and iteratively improving them. The user **"asks"** the API for a new set of 
hyperparameters to try and **"tells"** how these performed use a **"response"**. The API will maximize the response; if 
you want to minimize, just use the negative of the actual response.

## Basic example
This contains an example of how to tune the hyperparameters for a 
[random forest regression model from `scikit learn`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html). 
We focus on two hyperparameters but the example could be expanded to all the hyperparameters.

In the following we assume a user has been created and the token `<TOKEN>` has been provided. How to create a user and get a token is [detailed below](#Creating-a-user-and-getting-a-token).

### Step 1: Create experiment
Create an experiment with two variables
* `n_estimators`: an integer describing the number of trees in the forest
* `min_samples_split`: a continuous variable in the range [0; 1] giving the fraction of samples in each split.

```python
import requests

# url of the endpoint
create_url = "http://dev.autotune.localhost:8008/experiment/new"

# security: JWT header
headersAuth = {"Authorization": "Bearer <TOKEN>"}  # example token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0eXBlIjoiYWNjZXNzX3Rva2VuIiwiZXhwIjoxNjQ3ODQzMzAxLCJpYXQiOjE2NDcxNTIxMDEsInN1YiI6IjEifQ.h9r3zJ1RYZt7PoAvPpwne-MPIfKDNPsMq9nMmoRfiA8

# define experiment details, including the parameters we want to tune
json_data = {
  "name": "RandomForest_nestimators_minsampsplit",
  "description": "Two-parameter tuning of random forest",
  "covars": {
    "n_estimators": {
      "vtype": "int",
      "guess": 100,
      "min": 10,
      "max": 500
    },
    "min_samples_split": {
      "vtype": "cont",
      "guess": 0.3,
      "min": 0.0,
      "max": 1.0
    }
  },
  "model_type": "SingleTaskGP",
  "acq_func": "ExpectedImprovement"
}

# post to API
r = requests.post(create_url, headers=headersAuth, json=json_data)
```

The `covars`-part of `json_data` defines the hyperparameters of the experiment, with one dictionary per variable. [See 
further details below](#POST-action:-`experiments/new`-endpoint:-creating-a-new-experiment) of how to create experiments with more hyperparameters (and other data types).

## Creating a user and getting a token


# Endpoints
## Dev API
The dev API is orchestrated via `docker-compose.yml`. The API is available on `dev.autotune.localhost:8008` (the swagger 
entry point on `dev.autotune.localhost:8008/docs`), the `traefik` dashboard is available on `dev.autotune.localhost:8081` 
and the `pgadmin` tool for accessing the postgres database is available on `dev.autotune.localhost:5050` (user name and 
password available through `.env`-file)

## Example API calls

### `/auth/token` endpoint: Getting the access token
Example code for a user with username (email) `test@test.com` and password `CHANGEME`

In `python`
```python
import requests

url = 'http://dev.autotune.localhost:8008/auth/token'
data = {'username': 'test@test.com', 'password': 'CHANGEME'}

r = requests.post(url, data = data)
```
where the returned token can be grabbed via `r.json()["access_token"]`.

Using `curl` on the command line
```shell
svedel@svedel-T430s$ curl -X POST http://dev.autotune.localhost:8008/auth/token -H 'accept: application/json' -d 'username=test%40test.com&password=CHANGEME'
```

### POST action: `experiments/new` endpoint: creating a new experiment

For auth purposes this endpoints requires a JWT token to be submitted in the header. In the following it is assumed the 
token has already been obtained; section "`/auth/token` endpoint: Getting the access token" above explains how to 
retrieve the token.

Here we explain how to set up a new experiment with name `some_name` and description `some_description`. In addition, 
the variables for the experiment need to be specified
* Variable `v1` is an _integer_ in the range [0; 5] which we guess should take value 2
* Variable `color` is a _categorical_ variable with options `red`, `green` and `blue` which we guess should take value `red`
* Variable `weight` is a _continuous_ variable in the range [-3; 200] which we start at 6.6
* The model should be of type `SingleTaskGP` (default), and the acquisition function of type `ExpectedImprovement`

The covariates for the model (the variables) are defined via the `covars` object. Notice that options for categorical
variables (`color`) should be entered as a list in the json
```json
"covars": {
    "v1": {
      "vtype": "int",
      "guess": 2,
      "min": 0,
      "max": 5
    },
    "color": {
      "vtype": "cat",
      "guess": "red",
      "options": [
        "red", "green", "blue"
      ]
    },
    "weight": {
      "vtype": "cont",
      "guess": 6.6,
      "min": -3.0,
      "max": 200.0
    },
  }    
```
**Note I** that JSON format does not accept commas after last entries. 

**Note II** that the outcome options for categorical variables (`options` for the variable `color` in the example above) 
can only be processed by `FastAPI` if provided between square brackets `[]` instead of curly brackets `{}` (because this
field is defined as a set in `FastAPI`).

The full API call in `python` will look like this
```python
import requests

url = "http://dev.autotune.localhost:8008/experiment/new"  # use dev endpoint for API call
headersAuth = {"Authorization": "Bearer <TOKEN>"}  # example token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0eXBlIjoiYWNjZXNzX3Rva2VuIiwiZXhwIjoxNjQ3ODQzMzAxLCJpYXQiOjE2NDcxNTIxMDEsInN1YiI6IjEifQ.h9r3zJ1RYZt7PoAvPpwne-MPIfKDNPsMq9nMmoRfiA8

json_data = {
  "name": "some_name",
  "description": "some_description",
  "covars": {
    "v1": {
      "vtype": "int",
      "guess": 2,
      "min": 0,
      "max": 5
    },
    "color": {
      "vtype": "cat",
      "guess": "red",
      "options": [
        "red", "green", "blue"
      ]
    },
    "weight": {
      "vtype": "cont",
      "guess": 6.6,
      "min": -3.0,
      "max": 200.0
    }
  },
  "model_type": "SingleTaskGP",
  "acq_func": "ExpectedImprovement"
}
    
# post to API
r = requests.post(url, headers=headersAuth, json=json_data)
``` 
The API response will be available as `r.json()`. Again, notice how `options` for the `color`-variable are provided in
square brackets `[]` instead of curly brackets `{}`.

Using `curl` on the command line
```shell
svedel@svedel-T430s$ curl -X 'POST'   'http://dev.autotune.localhost:8008/experiment/new'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
>   "name": "some_name",
>   "description": "some_description",
>   "covars": {
>     "v1": {
>       "vtype": "int",
>       "guess": 2,
>       "min": 0,
>       "max": 5
>     },
>     "color": {
>       "vtype": "cat",
>       "guess": "red",
>       "options": [
>         "red", "green", "blue"
>       ]
>     },
>     "weight": {
>       "vtype": "cont",
>       "guess": 6.6,
>       "min": -3.0,
>       "max": 200.0
>     }
>   },
>   "model_type": "SingleTaskGP",
>   "acq_func": "ExpectedImprovement"
> }' -H "Authorization: Bearer <TOKEN>"

```

### GET action: `experiment/ask/{exp_uuid}` endpoint: get covariates for next experiment

For auth purposes this endpoints requires a JWT token to be submitted in the header. In the following it is assumed the 
token has already been obtained; section "`/auth/token` endpoint: Getting the access token" above explains how to 
retrieve the token.

Since this is a GET action, the endpoint is easier to call. In `python` the following will do the job (the experiment 
uuid `<EXP_UUID>` can be obtained from the `experiment/new` endpoint)
```python
import requests

url = "http://dev.autotune.localhost:8008/experiment/new/<EXP_UUID>"  # example exp_uuid: c111e09d-93fe-462c-bfba-3e72b727e52d
headersAuth = {"Authorization": "Bearer <TOKEN>"}  # example token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0eXBlIjoiYWNjZXNzX3Rva2VuIiwiZXhwIjoxNjQ3ODQzMzAxLCJpYXQiOjE2NDcxNTIxMDEsInN1YiI6IjEifQ.h9r3zJ1RYZt7PoAvPpwne-MPIfKDNPsMq9nMmoRfiA8

r = requests.get(url, headers=headersAuth)
r.json()

# will produce an output like the following if an experiment like the one in the example above is asked
# {'exp_uuid': 'c111e09d-93fe-462c-bfba-3e72b727e52d', 'time_updated': '2022-03-27T12:40:55.253019', 'covars_next_exp': '[{"v1":2,"color":"red","weight":6.6}]'}
```

## POST action: `experiment/tell/{exp_uuid}` endpoint: report results of experiment

This endpoint is used to report the outcome of an experiment, i.e. the response you got using the latest set of covariates
as proposed by the call to the `/experiment/ask/{exp_uuid}` endpoint.

The actual covariates used and the response obtained are provided as body parameters. They must be provided as two 
single-row `pandas` dataframes (one for covariates, another for response) converted to JSON using the `pandas` 
`to_json`-method.
* **Covariates**: use a separate column for each covariate and use the covariate name for the column. Data types used when defining the covariates must be followed (during experiment creation via `experiment/new`-endpoint via the `covars` parameter). 
* **Response**: use a single column named "Response"
The `pandas` `.to_json`-method can be used with either default options or with the `orient`-option specified.

This endpoint is secured by JWT token similar to other endpoints. 

In the following we extend the example from above and assume that the covariates are of the same (`v1` (type: `int`), `color` (type: `cat`) and `weigth` (type: `cont`))
```python
import requests
import pandas as pd

# results from experiment (obtained before calling API)
exp_response = pd.DataFrame({"Response": [6.2]})
exp_covars = pd.DataFrame({"v1": [3], "color": ["green"], "weight": [5.8]})

# data to post
euuid = "<EXP_UUID>"  # obtained from `/token` endpoint, example exp_uuid: c111e09d-93fe-462c-bfba-3e72b727e52d
tell_data = {"exp_uuid": euuid, "covars_tell": exp_covars.to_json(orient="records"), "response_tell": exp_response.to_json(orient="records")}  # the option orient="records" makes the JSON easier to read but is not necessary

# endpoint and JWT details
tellurl = "http://dev.autotune.localhost:8008/experiment/tell/" + euuid
headersAuth = {"Authorization": "Bearer <TOKEN>"}  # example token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0eXBlIjoiYWNjZXNzX3Rva2VuIiwiZXhwIjoxNjQ3ODQzMzAxLCJpYXQiOjE2NDcxNTIxMDEsInN1YiI6IjEifQ.h9r3zJ1RYZt7PoAvPpwne-MPIfKDNPsMq9nMmoRfiA8

# call API
r  = requests.post(tellurl, headers=headersAuth, json=tell_data)

# will produce an output like the following for an experiment like the one in the example above
# {'exp_uuid': 'c111e09d-93fe-462c-bfba-3e72b727e52d', 'covars_tell': "[{'v1': 3, 'color': 'green', 'weight': 5.8}]", 'response_tell': "[{'Response': 6.2}]", 'best_response': "{'Response': {'0': 6.2}}", 'covars_best_reponse': "{'v1': {'0': 3}, 'color': {'0': 'green'}, 'weight': {'0': 5.8}}", 'covars_sampled_iter': 1, 'response_sampled_iter': 1, 'time_updated': '2022-04-07T03:48:45.833469'}
```

## Prod API
See details under "Let's Encrypt" in tutorial from testdriven.io references below

# Security

Two different types of endpoint security has been implemented: sign-in based and token-based (JSON web token, JWT). Each supports a different experience flow: sign-in is for a website where the same user can be signed in over longer time, tokens are for an API product where the token is provided as part of the header to validate.

For token-based approaches, the code makes use of access tokens and refresh tokens. The basic idea is that access tokens
are signed and short-lived, while the non-signed and long-lived refresh token is the one which is used to issue the
access token. For more details see [this `StackOverflow` post](https://stackoverflow.com/questions/3487991/why-does-oauth-v2-have-both-access-and-refresh-tokens).

## Header-based validation (API product)

Implemented on all `experiment` endpoints, and for testing on the endpoint `/auth/header-me`

In this approach, the access token ( `<TOKEN>` obtained from `/auth/token` endpoint) is passed in the header 
of the API call. That is, in this flow, the token must be passed each time, but the user does not need to sign in first.
The user will need to have a token, but that's a one-off thing (and tokens can be refreshed, too). Example with the dev endpoint
```shell
vedel@svedel-T430s:~/fastapi_postgres_docker_check$ curl -X POST http://fastapi.localhost:8008/auth/header-me -H "Accept: application/json" -H "Authorization: Bearer <TOKEN>"
```

The backend checks token and raises exceptions if it's invalid. 

Details of this approach is given here: [Get started with FastAPI JWT authentication](https://dev.to/deta/get-started-with-fastapi-jwt-authentication-part-2-18ok)

The `fastapi` swagger page also supports manually providing JWT headers: for the endpoint you want to use, click on the padlock to the right to provide a valid token and then proceed to use the endpoint as you normally would.

## Login-based validation (approach for website)

For a website, we typically want users to sign in once and then just use the site dedicated to them (with their data 
etc). For this, a solution has been implemented in which users would log in once and then any subsequent API call would
reference the logged-in user. Endpoints:
* `/auth/login`: the endpoint to log in through (example of request message given here [Get started with FastAPI JWT authentication](https://dev.to/deta/get-started-with-fastapi-jwt-authentication-part-2-18ok))
* `/auth/me`: example of an endpoint that only returns user-based information after the user has logged in

Example call to `/auth/login` endpoint with user credentials `<USERNAME>` (email address) and `<PASSWORD>`
```shell
svedel@svedel-T430s:~/fastapi_postgres_docker_check$ curl -X POST http://fastapi.localhost:8008/auth/login -H "accept: application/json" -H "Content-type: application/x-www-form-urlencoded" -d "username=<USERNAME>&password=<PASSWORD>"
```

Example call to `/auth/me` endpoint to validate login
```shell
svedel@svedel-T430s:~/fastapi_postgres_docker_check$ curl -X GET http://fastapi.localhost:8008/auth/me
```

# Technical details

## Tech stack
* [`fastapi` API framework](https://fastapi.tiangolo.com/) with [`uvicorn` ASGI server](https://www.uvicorn.org/)
for a natively async API framework
* [`ormar` for API schemas and database models](https://collerek.github.io/ormar/) - natively async, integrates well
with `fastapi`, and is based on [`sqlalchemy`](https://www.sqlalchemy.org/) and 
[`pydantic`](https://pydantic-docs.helpmanual.io/) for API endpoint schemas and data models, and with [`alembic`](https://alembic.sqlalchemy.org/en/latest/) for database migration.
* [`postgres` database](https://www.postgresql.org/) with [`pgAdmin`](https://www.pgadmin.org/) for database management
* Networking productionized via [`træfik`](https://traefik.io/)
* `OAuth2` security via `JWT` tokens (JSON web tokens).
* Bayesian optimization library (doing the actual work): [`greattunes` for model tuning](https://pypi.org/project/greattunes/)

## Database management

An instance of the postgres admin tool `pgAdmin4` is set up for this application. For the development system it is 
available on port 5050 (`http://dev.autotune.localhost:5050/`). Login credentials are
* Username: "pgadmin4@pgadmin.org"
* Password: "admin"

Once logged in, the tool must be connected to the database. In the following we will use the development application 
database as example; more info also available in [this tutorial](https://ahmed-nafies.medium.com/fastapi-with-sqlalchemy-postgresql-and-alembic-and-of-course-docker-f2b7411ee396).  

Click the "Add New Server" link on the landing page, or right-click "Servers" in the tree on the 
left and select "Server" under "Create".

![alt text](docs/figs/pgadmin_create.png)

An entry box opens where you must first give the connection a name under the "General" tab. For this you can choose 
anything you want.

![alt text](docs/figs/db_connect.png)

Under the "Connection" tab you must add the details of the database containers. We are setting up the connection between 
two Docker containers, so we must use the Docker container service name and not the general address when connecting. In 
other words, the "Host name/address" should be set to the name of the postgres service in our Docker network; this name 
is `db`.

The "Maintenance database" name is the name of the database we want `pgadmin4` to connect to, i.e. the application 
database. For the development setup that name is `fastapi`. "Username" and "Password" are also to the same development
database, and the value for both are `fastapi` (these parameters are defined in `.env`).
 
Finally, the tables can be accessed using the tree to the left under Servers > autotune.dev > fastapi > schemas > tables

![alt text](docs/figs/find_tables.png)

# References
* [Christopher GS: blog on `fastapi` app building](https://christophergs.com/tutorials/ultimate-fastapi-tutorial-pt-10-auth-jwt/)
* [testdriven.io: Dockerizing FastAPI with Postgres, Uvicorn and Traefik](https://testdriven.io/blog/fastapi-docker-traefik/#postgres)
* [Get started with FastAPI JWT authentication](https://dev.to/deta/get-started-with-fastapi-jwt-authentication-part-2-18ok)
