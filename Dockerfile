# base image derivation 
FROM python:3.7-slim-stretch

# timezone handler 
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris 

# initial system requirements 
RUN apt-get update --fix-missing && \
    apt-get install --yes --no-install-recommends \
        tzdata apt-utils dialog gcc git curl pkg-config build-essential ffmpeg 

# user creation 
RUN useradd --gid root --create-home solver 
WORKDIR /home/solver 

# internal virtualenv 
ENV VIRTUAL_ENV=/opt/venv 
RUN chmod -R g+rwx /home/solver && python -m venv $VIRTUAL_ENV --system-site-packages 
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# copy requirements.txt 
COPY requirements.txt ./ 

# install python requirements 
RUN pip install --upgrade pip && pip install -r requirements.txt 
    
# pull source code 
COPY . ./ 

# entrypoint 
ENTRYPOINT ["python", "main.py"]
CMD ["--debug"]
