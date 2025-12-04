Digital Signal Processing Library.

### Project File Structure
```
.
├── main.py
├── README.md
├── processing
│   ├── __init__.py
│   ├── basic_ops.py
│   ├── generation.py
│   └── signal_digitize.py
...

```

## Local Setup
- clone the repo
```bash
git clone https://github.com/abdelrahman-omran/SignalFlow-Lab.git
```
- make sure you have python >= 3.10
    - using a virtual env is recommended
- install packages inside `requirements.txt` file

```bash
pip3 install -r requirements.txt
```

- make sure you are at the project's root directory, and run the app

```bash
python3 main.py
```

> or `python main.py` if on windows

### Testing Steps
**make sure you are at the project's root directory**

- execute the python file inside "tests/taskX" for the task you want to test
```bash
python3 tests/task1/*.py
```
