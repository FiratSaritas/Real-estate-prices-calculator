# Real-estate-prices-calculator

Where to find what:
- **Doc**: Contains our NB as PDF reports 
- **data**: Contains provided data for challenge
- **eda**: Notebooks for Exploratory Data Analysis 
- **model**: All notebooks and files regarding to our model building and submits for the Kaggle-Challenge 
- **util**: This folder includes several helper files
- **webinterface:** contains all files concerning our webinterface


-------------------------------------------------
# Immobilienpreisrechner: Description of the Project by Michael Graber

This is the repository for the Challenge Immobilienpreisrechner of the
Bachelor study program in Data Science at FHNW. The respective site on
ds-spaces can be found
[here](https://ds-spaces.technik.fhnw.ch/immobilienrechner/).

## Task

In this challenge we'll be investigating a Swiss real estate dataset with
machine learning methods to make predictions about object prices and types. 

You find the detailed description of the task on the [ds-spaces
site](https://ds-spaces.technik.fhnw.ch/immobilienrechner/).

## Dataset

The dataset can be found in the `data/`-directory. It has been collected by the
Information Processing Group of Prof. Dr. Manfred Vogel at the Institute of
Data Science.

Attribute names are mostly self explanatory. Here are some additional
explanations: Attributes starting with `gde_` refer to properties of the
municipality (Gemeinde). Attributes ending with `S`, `M`, `L` provide
information about the municipality and are sourced from the Bundesamt für
Statistik (BfS). The values correspond to mean values measured from areas of
various sizes, which correspond with the S, M and L attribute name endings.
Attributes starting with `gde_politics_` indicate shares of the corresponding
political parties in a given municipality.


## Development environment

Docker is a platform for developing and running applications and it's very
useful for developing Data Science codes in homogeneous environments as well.

Your submission at the end of the project will have to run inside the docker
container provided here.  After setting this up you'll be able to connect to
JupyterLab inside your docker container on your local machine through your
browser.

You are allowed to install additional packages. However, this needs to be
documented and reproducible through execution of a Jupyter Notebook, see ds-spaces.

Proceed as follows to install the development environment:


### 1. Install docker on your computer

Depending on your operating system you have to install docker in different ways.  

You'll find detailed instructions here: https://docs.docker.com/get-docker


### 2. Pull the challenge docker image

Authenticate to the GitLab Docker registry with a personal access token, see
details
[here](https://docs.gitlab.com/ee/user/packages/container_registry/#authenticating-to-the-gitlab-container-registry).

Choose scope `read_registry` und `write_registry` to generate the access token.  

```
# login to the FHNW GitLab docker registry first
$ docker login cr.gitlab.fhnw.ch -u <username> -p <token>

# now pull the image
$ docker pull cr.gitlab.fhnw.ch/ml/sgds/challenges/immobilienpreisrechner:v20200914
```

### 3. Fork this repository

Fork this repository to your own user space by pressing the fork button on the
upper right on this Repos GitLab page.

Add @michael.graber as a maintainer to your fork. If you don't do this I won't see your submission.

As a team you can jointly work on one fork. To figure out how do this best, have a look at  
the 'Kompetenz Software Konstruktion',
[here](https://gitlab.fhnw.ch/jasmin.fluri/softwarekonstruktion-data-science/-/blob/master/lessons/le1.md)
on GitLab.

I would recommend not to jointly work on the same Jupyter Notebooks. These are typcally complicated to merge.


### 4. Clone your fork to your computer. 

For this you might wanna set up a ssh-key for your computer, see here: https://docs.gitlab.com/ee/ssh/

In your fork on GitLab find the address you can clone your Repo with and execute:

```
$ git clone MY_REPO_FORK
```


### 5. Start a challenge container on your machine

```
$ docker run -d \
    -p 8866:8888 \
    --user root \
    -v PATH_TO_MY_REPO:/home/jovyan/
    --name=immo_challenge \
    cr.gitlab.fhnw.ch/ml/sgds/challenges/immobilienpreisrechner:v20200914 start.sh jupyter lab --LabApp.token=''

```

### 6. Check that your container is running

```
$ docker ps -a
```

### 7. Connect to your container through your browser

Enter `http://localhost:8866/lab` in your browser.


### 8. Restart

If you later on need to restart your container you can just run

```
$ docker start immo_challenge
```


## Submission 

You can now edit files in the clone of your fork on your local machine.
Whenever you want to upload something to your Fork on GitLab you need to
'commit' and 'push' it:

```
# commit MY_FILE
$ git commit MY_FILE -m 'my commit message'

# push all commits to the server
$ git push
```

**By pushing your commits to your fork on GitLab you will ultimately submit
your challenge. The last commit before the deadline is your submission.**

