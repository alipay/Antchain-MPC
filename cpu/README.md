### Step 1: Install requirements:

`python3 -m venv stf_venv`

`source ./stf_venv/bin/activate`

`python3 -m pip install --upgrade pip`

`pip3 install -r requirements.txt`


### Step 2: Edit the config file:

edit the file `./conf/config.json`  and  `./conf/config_linear.json`
Modify the following segment (please use three unused ports and use absolute path)：
```
    "hosts": {
        "workerL": "127.0.0.1:8886",
        "workerR": "127.0.0.1:8887",
        "RS": "127.0.0.1:8888"
    },
    "stf_home": "/..../Antchain-MPC/cpu",   
    "stf_home_workerL": "/..../Antchain-MPC/cpu",  
    "stf_home_workerR": "/..../Antchain-MPC/cpu",    
    "stf_home_RS": "/..../Antchain-MPC/cpu",      
```


### Step 3: Run the experiments in the paper.

`bash run.sh`

