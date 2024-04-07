# Pipe-NN

A tiny experiment of a shell-composable neural network.

I thought about composing NN layers with processes for each layer piped into one another in a shell, like:

```shell
$ cat input_data.txt | input | hidden --neurons 5 --activation sigmoid | hidden --neurons 3 --activation relu | output
```

## How to run

There are hardcoded neuron counts and weights to solve the XOR problem. I got the weights from a [stack exchange answer](https://ai.stackexchange.com/questions/6167/what-is-the-best-xor-neural-network-configuration-out-there-in-terms-of-low-erro). The `data.txt` file contains the XOR inputs.

```shell
$ export PATH="$PATH:$(pwd)/target/debug"
$ cargo build
$ cat data.txt | input | hidden | output
```

## Considerations

- This is definitely not production code;
- I still haven't figured out backpropagation, i'll need some kind of IPC to do the backwards flow in the pipeline.
