# ATC
Automatic Text Categorization

## Run Tests
```
python -m source.test
```

How to Create Envirionment (macOS)

## Install Homebrew
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bash_profile
exec $SHELL
brew doctor
```
## Install Pyenv
```
brew install pyenv
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
exec $SHELL
pyenv install 3.7.0
```
## Pyenv-virtualenv
```
brew install pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
exec $SHELL
```
## Create Virtual Environment
```
pyenv virtualenv 3.7.0 ATC
```
## Activate EHSI Envirionment
```
pyenv activate ATC
```
# Upgrade pip
```
python -m pip install --upgrade pip
```
## Install Requirements
```
pip install -r requirements.txt
```

How to Create Envirionment (Windows)

## Install Anaconda

## Create Conda Envirionment
```
conda create --name ATC python=3.7.0
```
## Activate EHSI Envirionment
```
activate ATC
```
# Upgrade pip
```
python -m pip install --upgrade pip
```
## Install Requirements
```
pip install -r requirements.txt
```