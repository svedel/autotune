# Basics

Natively async API for productionizing data science models, with load balancing and a separate production setup.

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
The dev API is orchestrated via `docker-compose.yml`. The API is available on `fastapi.localhost:8008` (the swagger 
entry point on `fastapi.localhost:8008/docs`), the `traefik` dashboard is available on `fastapi.localhost:8008` 
and the `pgadmin` tool for accessing the postgres database is available on `fastapi.localhost:5050` (user name and 
password available through `.env`-file)

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

## References
* [Christopher GS: blog on `fastapi` app building](https://christophergs.com/tutorials/ultimate-fastapi-tutorial-pt-10-auth-jwt/)
* [testdriven.io: Dockerizing FastAPI with Postgres, Uvicorn and Traefik](https://testdriven.io/blog/fastapi-docker-traefik/#postgres)
* [Get started with FastAPI JWT authentication](https://dev.to/deta/get-started-with-fastapi-jwt-authentication-part-2-18ok)