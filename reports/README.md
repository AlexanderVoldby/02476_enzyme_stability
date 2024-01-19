---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages *
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [x] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [x] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [x] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [x] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

--- 40 ---

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

--- s214591, s222856, s216708, s214633 ---

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- We used the Transformers framework to easily run a pre-trained BERT-based protein model for tokenizing and embedding our data. Initially, we download the model and tokenizer from Huggingface. We then save it locally, to more quickly load it for inference.  We used Pytorch-Lightning to simplify training, prediction, and saving/loading of checkpoints. Weights and Biases was used for logging together with Pytorch-Lightning. Therefore, we were able to save checkpoints in a Google Bucket as well as through the Huggingface workspace. ---

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:


--- To handle Python dependencies, we specified two requirements files. They were manually created throughout the project. 
To recreate the environment, one would clone the repository, create a conda environment with the correct python version and install the requirements with pip.
For example:
- git clone https://github.com/AlexanderVoldby/02476_enzyme_stability.git
- conda create --name enzyme_stability python=3.11
- conda activate enzyme_stability
- pip install -r requirements.txt
- pip install -r requirements_dev.txt ---


### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

--- 
From the cookiecutter template we have filled out the data, models, reports, mlops_enzyme_stability (source code), mlops_enzyme_stability/data and mlops_enzyme_stability/models/ folders. In addition to that we also used the folder for the dockerfiles for both our train and predict dockerfiles. We created a folder called "tests" for our unittesting. We removed the notebooks, docs and mlops_enzyme_stability/visualizations since we did not use them. The different config files were placed in the root directory of the project. ---

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

--- We did not implement a concrete set of rules for code quality and format, since the lifespan of the project was rather short and we were able to explain the code to each other in person. But a consistent coding format matters a lot in larger projects because you will be dependent on understanding the code, someone else wrote, and vice versa. This is particularly the case for debugging purposes, when functions and classes interoperate. ---


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

--- We implemented two test functions that each run 2-3 tests. One function tests that our model output has the right format and that the model uses the correct loss function. The data testing function asserts that preprocessed data points have the correct format and that tensors are stored in an accessible path. ---


### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- 
Our code coverage is of 63%. Even if we had a code coverage of 100%, we still could expect finding some errors. Code coverage only shows how much of our code is 
executed when running the tests. However, it does not cover all possible scenarios our code could be run, for example in terms of the diversity of our inputs, we will always be limited by the exhaustive component of our tests. 

---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

--- Branches were an important part of our workflow. Everyone working on the codebase, had their own branch. Later on these branches would be merged again with the master branch using pull requests. Since this was done quite frequently was there no branch protection on the master branch, so we relied on everyone to go through their own pull request with care. Later on when working on the continuous integration and the automatic cloud deployment we used the main branch for deployment. In that way could the master branch be used for development, and each time we would like to deploy the current state of the codebase a pull request from the master to the main branch was created. This pull request then had to be reviewed by at least 2 people to include more safety before deployment. ---

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

--- In our project we did use DVC for managing our data. Unlike in most other project, training our model was not the computationally most intense part of the pipeline. It was in fact the preprocessing in the form of embedding the amino acid sequences. DVC helped us to make sure, that the already pre-processed data could easily be pulled using DVC. This also ensured that everyone was working with the same state of pre-processed data. If at any point we would have changed our pre-processing pipeline the data could be updated with DVC. Later on, we stopped relying on DVC and instead pulled data directly from Google Buckets. ---

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

--- 
We relied on unittesting in the form of pytest for our  continuous integration setup. It is organized into three different test scripts, on focusing on data preprocessing, on focusing on the model itself and on focusing on making predictions using the model. As we are mostly working on Windows and Linux did the testing include the latest ubuntu and windows distribution. The workflow for testing is triggered every time someone pushes to master or main and does a pull-request to these two branches. We utilized the caching option in GitHub to reduce the time it takes to run every workflow, since GitHub restricts the amount of time run for workflows. An example of a workflow can be found here: https://github.com/AlexanderVoldby/02476_enzyme_stability/actions/runs/7540382637 

---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

--- 
Our approach for configuring experiments and keeping track of past configurations was a `config.yaml` file with hydra. The config files contains various hyperparameters of the model, but also other options as the name of the run. In that way a the model can be run in the terminal as:
``$ python mlops_enzyme_stability/train_model.py hyperparameters.lr=0.002 hyperparameters.epochs=10 runname=”Run_1”``
In this example the learning rate and epochs as well as the name of the run are specified explicitly. ---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- In addition to keeping track of config files using hydra we tracked all of our experiments using W&B. This way all team members have easy access to all of the runs and secures the configuration and experiments in an accessible cloud format. To enable actual reproducibility everything must be seeded to enable deterministic behaviour. For this we utilized the seeding function of Pytorch Lightning, as this sets the seed all the relevant random number generators we are using. To reproduce a run one could load the corresponding hydra config files, which by default are saved in the “outputs” folder. Alternatively, one could also use the logging function from Pytorch Lightning, which saves configuration in the lightning_logs folder. As the config.yaml contains the Pytorch Lightning seed, this allows for reproducible experiments. ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- ![Local Image](figures/table_wandb.png) ![Local Image](figures/graphs_wandb.png)
The main metric we tracked for all of our experiments was the training loss. Besides that the number of epochs are tracked as well as all the hyperparameters from the config.yaml file. The training loss was tracked, because it is the first metric to use when evaluating if the model is actually learning the data set. After doing a set of grid-based experiments (changing the batch size and number of nodes for each layer) it could be seen that the model fundamentally struggled learning the training data set. Since this could not be solved by adjust hyperparameters as the learning rate or the batch, we suspect that there could be a problem in embedding our data that leads to a suboptimal training set. As we focused more on building a robust cloud based pipeline for our model we did not use more time optimizing the data preprocessing. If the project would go on more time could be allocated to resolving the issues with the preprocessing. After implementing this it would also be of interest to track the testing error to evaluate if the model is overfitting or not generalizing enough. As the embedded data set is rather small training the model is not computationally extensive. Therefore, using sweeps a large number of experiments could be run to find the optimal configuration of hyperparameters. ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- In our project we developed docker images for the training of our model and for making predictions with our model. For example, for training the model one could run the docker image with the following command, specifying the learning rate and the path to the data.
``$ docker run --name experiment1 trainer:latest lr=0.005 data_path:="gs://protein_embeddings/data/processed"``
To run the Docker image containing our prediction API locally we would laso specify a port to host it locally:
"$ docker run -p 8000:8000 prediction:latest"
Later on docker images where mostly build and run in gcloud as part of our continuous integration pipeline.
Link to docker file: ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- For debugging, we gave each team member the freedom to choose their tools of choose. Most of the time this was the debugging functionality in VS Code. In addition to that, error were looked up with Stack Overflow. With more difficult cases, ChatGPT could also be helpful to get more ideas of what could be the underlying issue.
Regarding profiling, our model was rather small, so training did not take much resource. In addition to that we tried to make our could lean using Pytorch Lightning. For these reasons was profiling not on of our priorities, as we thought that building a robust pipeline for training and predictions in the cloud was more important. ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- We used the following services:
- Cloud Storage: We defined and used a bucket to store the training data for our project as well as the model checkpoint, which is retrieved via Data Version Control operations. Cloud storage is a platform that allows to store and retrieve data in a structured way. It organizes the data into buckets, which could be interpreted as folders. Data Version Control was implemented via dvc.
- Cloud Build: We used Cloud Build to automatically build two Docker containers upon trigger: one for training our model, and the other to generate the predictions and configure the API. Cloud Build is a continuous integration continous delivery platform that facilitates the building of applications from source code, in our case two Docker containers, and integrates with other Google Cloud servcies such as Cloud Run.
- Cloud Run: We used Cloud Run to make our Cloud Build API accessible without any server. Cloud Run is a serverless computer platform that allows running containerized applications and make them accessible through HTTP requests. In addition, Cloud Run is integrated with Cloud Build as part of the continuous integration pipeline. However we did not manage to automatize the integreation between Cloud run and Cloud Build, so we manually selected the image from the Artifact Registry with Cloud Run to make the API accessible.
- Vertex AI: We used Vertex AI to build and manage the development of our model, for example we used to explore the right hyperparameters. Vertex AI is machine learning tailored platform for development and management of a machine learning workflow at all steps. --- 

### Question 18 CHECK 

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- We used the compute engine virtual machines to host the training and predicting Docker containers. 
The training and predictions VMs had the following hardware: 
 - Machine type: n1-highmem-2
 - CPU platform: Intel Haswell
 - Minimum CPU platform: None
 - Architecture: -
 - vCPUs to core ratio: - 
 - Custom visible cores: - 
 - GPUs: None

The training Docker container specifications:
```
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY mlops_enzyme_stability/ mlops_enzyme_stability/
COPY data/ data/
COPY config.yaml config.yaml

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_dev.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "mlops_enzyme_stability/train_model.py"]```
```
The API predictions Docker container specifications:

```
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_enzyme_stability/ mlops_enzyme_stability/
COPY data/ data/
COPY config.yaml config.yaml

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN python mlops_enzyme_stability/data/download_BERT.py

EXPOSE 8080

CMD ["uvicorn", "mlops_enzyme_stability.predict_sequence:app", "--host", "0.0.0.0", "--port", "8080"]
``` 
---


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:
--- 
![Local Image](figures/buckets_registry.png)
![Local Image](figures/bucket_content.png) --- 


### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- ![Local Image](figures/container_registry.png) ---


### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- ![Local Image](figures/cloud_build_history.png) ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:
---
To deploy our model, we wrapped our model into an API using FastAPI in a Docker contaier. The API allows inserting the sequence of aminoacids and generate the protein stability predictions. We initially deployed the model locally, which worked. Subsequently, we deployed our model in the cloud using Cloud Build to build the docker container and deployed it with Cloud Run. To invoke it an user would call `curl -X POST -F "file=@file.json" https://predict-model-nvnqcfzrxa-ew.a.run.app/predict/` (might be deleted). The input to the model is a list of amino acid sequences in string format. The response is a list of predicted stability values. ---


### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- So far we have not implemented monitoring. In short, monitoring would prevent the degradation of our application. In our case, a main thread for our application's performance would be data drifting, when for example, better protBert version is released, which make our application less competitive, or if a greater amount of training sequences becomes available, which would upgrade our application generalization potential. We would like to monitor the performance of our model over the protein sequences available. ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:
---
In total, 223.81 kr were spend on the project. About 95 % of the cost was due to cloud storage. This was probably due to the amount of container images and data stored in the cloud. Training the model via Vertex AI just cost around 8 kr. The cost for the compute engine was neglectable. ---

CONTINUE WITH Coding environment

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- 
To explain the overall architecture of our project, we can do it from three perspectives: the data and model retrieval, the project development and the user perspective:
- Data and model retrieval: Starting from the bottom left diagram, our project started by retrieving the raw Datafiles from Kaggle containinf the protein embeddings, preproces and store them in a Google Cloud Storage Bucket, we then downloaded the pretrained protBERT model from Huggingface, which would be used at the inference time to generate embeddings from new protein sequences.
- Project development: Once the model was implemented, executable, and the best model was stored as checkpoint in the Google Cloud Storage Bucket, we implemented a set of GitHub actions so every time we pushed changes to master, unit tests where executed. Pushing changes to the master branch also triggered Google Cloud Build to build two Docker images that included the last changes, one image executed the training of the model, and the other image hosted the API to generate predictions. Both containers were registered in the Google Cloud Artifact Registry. Google Cloud Run was then used to serve the API hosted by the prediction Docker container, whereas we used Vertex AI to explore and optimize the training performance, in both cases the Docker images were retrieved from the Artifact Registry. We used Weights and Biases to track the training metrics when running models on Vertex AI, and checkpoint were automatically saved on the Bucket. 
- User perspective: Once the API served by Cloud run and accessible, the protein stability can be predicted from the aminoacid sequence inserted.

![Local Image](figures/Architecture_Enzyme_Stability.jpg) 

---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:
---
A lot of problems arose from the path structure in our project, such as accessing the appropriate configuration files during both training and inference. This simply took a lot of debugging to overcome. We also faced some challenge with authentication of Google Cloud and Wandb when running Docker containers. This was solved partly by using private keys and by making our GCS budkets public. Docker was also a challenge initially as weinitially weren't accustomed to it and spent a lot of time waiting for Docker images to be built in order to run them.

---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- 
- 
Student s228562 was in charge of continous integration including unittesting and GitHub Actions, doing training runs varying the hyperparameters in a grid-based manner and contributing to the code base for the data, training and prediction.
- Student s214591 was primarily in charge of coding the prediction API, setting up automatic building and pushing of containers to the cloud, and deploying the prediction API in Cloud Run. Apart from this he set up the initial Cookicutter project and worked on setting up the training script and applying the pre-trained model for data processing.
- Student s214633 setup logging with Weights and Biases, handling of API keys with Hydra arguments for security, implemented Pytorch Lightning, performed the preprocessing of the raw data on DTU's HPC and helped setup training on GCP.
- Student s216708 contributed on setting up the data version control, the Google Cloud project creation and accessibility, set up the Google Cloud Bucket and the data/models saving and retrival from it. He also helped on the overall project design. 

---
