# AutoTune

Natively async API for Bayesian optimization-based model tuning, with load balancing and a separate production setup.

## Tech stack
* [`fastapi` API framework](https://fastapi.tiangolo.com/) with [`uvicorn` ASGI server](https://www.uvicorn.org/)
for a natively async API framework
* [`ormar` for API schemas and database models](https://collerek.github.io/ormar/) - natively async, integrates well
with `fastapi`, and is based on [`sqlalchemy`](https://www.sqlalchemy.org/) and 
[`pydantic`](https://pydantic-docs.helpmanual.io/) for API endpoint schemas.
* [`postgres` database](https://www.postgresql.org/) with [`pgAdmin`](https://www.pgadmin.org/) for database management
* Networking productionized via [`træfik`](https://traefik.io/)
* `OAuth2` security via `JWT` tokens (JSON web tokens).
* your favorite data science library (to be added soon: [`greattunes` for model tuning](https://pypi.org/project/greattunes/))

## Endpoints
### Dev API
The dev API is orchestrated via `docker-compose.yml`. The API is available on `dev.autotune.localhost:8008` (the swagger 
entry point on `dev.autotune.localhost:8008/docs`), the `traefik` dashboard is available on `dev.autotune.localhost:8081` 
and the `pgadmin` tool for accessing the postgres database is available on `dev.autotune.localhost:5050` (user name and 
password available through `.env`-file)

### Example API calls

#### `/auth/token` endpoint: Getting the access token
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

#### POST action: `experiments/new` endpoint: creating a new experiment

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

#### GET action: `experiment/ask/{exp_uuid}` endpoint: get covariates for next experiment

For auth purposes this endpoints requires a JWT token to be submitted in the header. In the following it is assumed the 
token has already been obtained; section "`/auth/token` endpoint: Getting the access token" above explains how to 
retrieve the token.

Since this is a GET action, the endpoint is easier to call. In `python` the following will do the job (the experiment 
uuid `exp_uuid` can be obtained from the `experiment/new` endpoint)
```python
import requests

url = "http://dev.autotune.localhost:8008/experiment/new/<EXP_UUID>"
headersAuth = {"Authorization": "Bearer <TOKEN>"}  # example token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0eXBlIjoiYWNjZXNzX3Rva2VuIiwiZXhwIjoxNjQ3ODQzMzAxLCJpYXQiOjE2NDcxNTIxMDEsInN1YiI6IjEifQ.h9r3zJ1RYZt7PoAvPpwne-MPIfKDNPsMq9nMmoRfiA8

r = requests.get(url, headers=headersAuth)
r.json()

# will produce an output like the following if an experiment like the one in the example above is asked
# {'exp_uuid': '6d11cdf0-8b13-4140-839d-1f924653589b', 'time_updated': '2022-03-27T12:40:55.253019', 'covars_next_exp': '[{"v1":2,"color":"red","weight":6.6}]'}
```


### Prod API
See details under "Let's Encrypt" in tutorial from testdriven.io references below

## Security

Three different types of endpoint security has been implemented. Each supports a different experience flow, so
implementing all 3 is about learning.

For token-based approaches, the code makes use of access tokens and refresh tokens. The basic idea is that access tokens
are signed and short-lived, while the non-signed and long-lived refresh token is the one which is used to issue the
access token. For more details see [this `StackOverflow` post](https://stackoverflow.com/questions/3487991/why-does-oauth-v2-have-both-access-and-refresh-tokens).

### Header-based validation (API product)

Implemented on the endpoint `/auth/header-me`

In this approach, the access token ( `<TOKEN>` obtained from `/auth/token` endpoint) is passed in the header 
of the API call. That is, in this flow, the token must be passed each time, but the user does not need to sign in first.
The user will need to have a token, but that's a one-off thing (and tokens can be refreshed, too). Example with the dev endpoint
```shell
vedel@svedel-T430s:~/fastapi_postgres_docker_check$ curl -X POST http://fastapi.localhost:8008/auth/header-me -H "Accept: application/json" -H "Authorization: Bearer <TOKEN>"
```

The backend checks token and raises exceptions if it's invalid. 

Details of this approach is given here: [Get started with FastAPI JWT authentication](https://dev.to/deta/get-started-with-fastapi-jwt-authentication-part-2-18ok)

### Login-based validation (approach for website)

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

### HTTP validation 

## Support tools

### Database management

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

## References
* [Christopher GS: blog on `fastapi` app building](https://christophergs.com/tutorials/ultimate-fastapi-tutorial-pt-10-auth-jwt/)
* [testdriven.io: Dockerizing FastAPI with Postgres, Uvicorn and Traefik](https://testdriven.io/blog/fastapi-docker-traefik/#postgres)
* [Get started with FastAPI JWT authentication](https://dev.to/deta/get-started-with-fastapi-jwt-authentication-part-2-18ok)