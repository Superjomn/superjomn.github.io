+++
title = "Enable Jupyter in Doom Emacs"
author = ["Chunwei Yan"]
date = 2024-11-02
tags = ["tech", "emacs"]
draft = false
+++

There are a few adjustments needs for the default installation when using the [jupyter package](https://github.com/emacs-jupyter/jupyter) in Emacs. Here's a step-by-step guide to configure it properly with Doom Emacs.


## Step 1: Install the jupyter package. {#step-1-install-the-jupyter-package-dot}

Add this line to `package.el`:

```emacs-lisp
(package! jupyter)                      ;
```


## Step 2: Enable builtin Jupyter Support in Org Mode {#step-2-enable-builtin-jupyter-support-in-org-mode}

To enable Jupyter support in Org mode, make the following modifications in your `init.el` file:

1.  Uncomment the `ein` line. The [emacs-ipython-notebook](https://github.com/millejoh/emacs-ipython-notebook) is a dependency of jupyter package.
2.  Add `+jupyter` to the Org settings. For more details, refer to [:lang org](https://github.com/doomemacs/doomemacs/blob/5dcba2f89fa5a20c6535e15f859aaef466ce4b90/modules/lang/org/README.org#L63):

<!--listend-->

```emacs-lisp
(org +jupyter)               ; organize your plain life in plain text
```


## Step 3: Patch for Runtime Errors with ZeroMQ {#step-3-patch-for-runtime-errors-with-zeromq}

To address a runtime error related to ZeroMQ (as discussed in this [issue](https://github.com/emacs-jupyter/jupyter/issues/527#issuecomment-2391691176)), append the following code to your `config.el` or any other configuration file:

```emacs-lisp
(defun my-jupyter-api-http-request--ignore-login-error-a
    (func url endpoint method &rest data)
  (cond
   ((member endpoint '("login"))
    (ignore-error (jupyter-api-http-error)
      (apply func url endpoint method data)))
   (:else
    (apply func url endpoint method data))))
(advice-add
 #'jupyter-api-http-request
 :around #'my-jupyter-api-http-request--ignore-login-error-a)
```


## Step 4: reload the Emacs {#step-4-reload-the-emacs}

After making these changes, reload Emacs. Doom Emacs should now install the necessary packages.


## Using the Jupyter Package {#using-the-jupyter-package}

To use the jupyter package, you can start an `org` file, and invoke `jupyter-run-repl` command, it will start a kernel for this file. The python code blocks will map to the jupyter session.

For example:

```python
2 ** 10
```

```text
1024
```
