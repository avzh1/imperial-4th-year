# Setting up a Python virtual environment

```shell
python3 -m virtualenv [your-path]
# activate virtual environment
source [your-path]/bin/activate
# install requirements from requirements.txt
pip install --upgrade pip
pip install -r $DIR/requirements.txt
# Exit with command `deactivate`
```