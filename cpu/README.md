### Step 1: Download the code:

`git clone https://github.com/alipay/Antchain-MPC.git`

`cd Antchain-MPC`

`git checkout -b sec_softmoid origin/sec_softmoid`

`cd cpu`




### Step 2: Install requirements:

`python3 -m venv stf_venv`

`source ./stf_venv/bin/activate`

`python3 -m pip install --upgrade pip`

`pip3 install -r requirements.txt`


### Step 3: Edit the config file:

edit the file `./conf/config.json`  and  `./conf/config_linear.json`
Modify the following segment：
```
    "hosts": {      # use three unused ports
        "workerL": "127.0.0.1:8886",
        "workerR": "127.0.0.1:8887",
        "RS": "127.0.0.1:8888"
    },
    "stf_home": "/..../Antchain-MPC/cpu",            #  use absolute path
    "stf_home_workerL": "/..../Antchain-MPC/cpu",    #  use absolute path
    "stf_home_workerR": "/..../Antchain-MPC/cpu",    #  use absolute path
    "stf_home_RS": "/..../Antchain-MPC/cpu",       #  use absolute path
```


### Step 4: Run the experiments in the paper.

`bash run.sh`

