# FMEngine CLI

**Usage**:

```console
$ fmengine [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `eval`
* `export`
* `inference`
* `prepare-ckpt`
* `train`

## `fmengine eval`

**Usage**:

```console
$ fmengine eval [OPTIONS]
```

**Options**:

* `--model-id TEXT`: Path to the model file  [required]
* `--tasks TEXT`: Comma-separated tasks to evaluate  [required]
* `--help`: Show this message and exit.

## `fmengine export`

**Usage**:

```console
$ fmengine export [OPTIONS]
```

**Options**:

* `--ckpt-path TEXT`: Path to the checkpoint file  [required]
* `--step INTEGER`: Step to export the model  [default: -1]
* `--config TEXT`: Path to the config file  [required]
* `--output-path TEXT`: Path to the output directory  [required]
* `--help`: Show this message and exit.

## `fmengine inference`

**Usage**:

```console
$ fmengine inference [OPTIONS]
```

**Options**:

* `--model-id TEXT`: Path to the model file  [required]
* `--prompt TEXT`: Prompt to generate text  [required]
* `--temperature FLOAT`: Temperature for sampling  [default: 0.5]
* `--top-k INTEGER`: Top k for sampling  [default: 50]
* `--top-p FLOAT`: Top p for sampling  [default: 0.9]
* `--revision TEXT`: Revision of the model
* `--help`: Show this message and exit.

## `fmengine prepare-ckpt`

**Usage**:

```console
$ fmengine prepare-ckpt [OPTIONS]
```

**Options**:

* `--config TEXT`: Path to the config file  [required]
* `--help`: Show this message and exit.

## `fmengine train`

**Usage**:

```console
$ fmengine train [OPTIONS]
```

**Options**:

* `--config TEXT`: Path to the config file  [required]
* `--help`: Show this message and exit.
