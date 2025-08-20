1. brew install pyenv pyenv-virtualenv


Load pyenv automatically by adding
the following to ~/.bashrc:

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

1. pyenv 3.13
1. pyenv global 3.13