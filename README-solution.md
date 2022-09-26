# MAZE CLI SOLVER
This tools is an automatic maze generator and solver
It's based on numpy and click 

# code organization 
* librairies
    * log.py
    * this file defines the logger 
    * strategies.py
    * this file is the core of the app
    * it contains all strategies such as maze-generator, heuristic_search
* requirements.txt
* Dockerfile
* main.py
    * this file is the entrypoint of the app
    * it exposes two subcommands : create and solve 
* .dockerignore
* .gitignore

# requirements 
* python3
* venv 
* pip 

# initialization
```bash
# create virtual env
python -m venv env 
# activate virtual env
source env/bin/activate
# update pip  
pip install --upgrade pip
# install python dependencies from requirements 
pip install -r requirements.txt 
```

# run python script 
```bash
# main help
python main.py --help 
# maze create help 
python main.py create --help 
# maze solve help
python main.py solve --help
# create maze
python main.py create --height 11 --width 11 --seed 190876
# create and solve maze 
python main.py create --height 11 --width 11 --seed 190876 | python main.py solve
# solve maze from file
cat unsolvable_maze.txt | python main.py solve 
```

# build docker image 
* build image 
```bash
# build image 
docker build -t mazemap:0.0 -f Dockerfile .
``` 

# run docker container 
```bash
# main help
docker run --rm mazemap:0.0 --help
# maze create help 
docker run --rm mazemap:0.0 create --help
# maze solve help
docker run --rm mazemap:0.0 solve --help
# create maze 
docker run --rm --name maze-builder --tty mazemap:0.0 create --height 21 --width 21 --seed 190987
# create and solve maze
docker run --rm --name maze-builder --tty mazemap:0.0 create --height 21 --width 21 --seed 190987 | docker run --rm --name maze-solver --interactive mazemap:0.0 solve
# solve maze from file
cat unsolvable_maze.txt | docker run --rm --name maze-solver --interactive mazemap:0.0 solve
```
