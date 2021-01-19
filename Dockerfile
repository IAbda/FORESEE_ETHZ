# Performance note: python:3.8 is quite a large container, because it is based on the full linux distribution,
# most of which will never get used. This:
#  - takes up a lot of space on your hard drive, which is fairly bad, and
#  - takes a lot of time to deploy, which is very frustrating when doing devops and CI/CD, and
#  - (in serverless environments where memory <> hard drive space) costs a *lot* to be running as you're paying for
#    hundreds of mb of unused capacity.
# Using one of the slim-buster images (or if you can get away with it, which if you're using sklearn you can't,
# slim-alpine) will save several hundred mb of disk space.
#
# You can investigate this by doing "docker image ls -a"
#   python                     3.8.7-slim-buster     be5d294735c6   2 days ago       113MB
#   python                     3.8                   b0358f6298cd   2 days ago       882MB

FROM python:3.8.7-slim-buster

# Style note: I've altered /FORESEE_ETHZ_APP/ to /foresee_ethz_app/ to eliminate potential confusion that on first
# glance, this seems like an environment variable name (all caps by convention)

RUN mkdir -p /foresee_ethz_app/

WORKDIR /foresee_ethz_app/

COPY requirements.txt .

# Style Note: Used the canonical version of pip, rather than pip3. It'll be pathed to the correct python version, and
# I'm not sure if pip3 will continue to be officially supported now that python2.7 is deprecated formally.

RUN pip install -r ./requirements.txt

COPY . .

# Style Note: Same, there should be only one version of python (3.8.7) pathed so if there isn't that's a good clue that
# something's wrong further up

CMD ["python", "./train_traffic_intensity_RF_model.py"]
