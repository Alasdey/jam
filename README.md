# jam
A jam trying to reach escape velocity.

## To use
First try to recreate the missing rewards folder as it will not function without. \
Also compile the subleq interpreter:
```bash
gcc -shared -fPIC -o interpreters/subleq/libsubleq.so interpreters/subleq/subleq.c
python3 main.py
```

## Todo's
Make a wrapper for the interpreters to allow for more UISC or RISC languages (be it marginal improvements). \
Make a wrapper for the rewards \
Build a logging system \
Improve the payoff computation \
Import the Nash set computation \
Import the Genetic Programming 
