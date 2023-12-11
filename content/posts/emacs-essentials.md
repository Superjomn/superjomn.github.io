+++
title = "Emacs Essentials"
author = ["Chunwei Yan"]
date = 2022-10-15
tags = ["emacs", "tech"]
draft = false
+++

It is a steep learning curve to master Emacs lisp, there are mainly two issues in it from my experience

1.  the lisp syntax and functional programming
2.  the fragmented methods and libraries

For the 1st issue, it is easy to master the syntax after writing several programs and getting used to them, but for the 2nd one, one needs to take notes or remember something.

In this blog, I focus on the 2nd point and keep updating the notes of some methods and libraries that I think are essential for writing Emacs lisp packages.


## builtin methods {#builtin-methods}


### buffer {#buffer}


#### current-buffer: get the current buffer {#current-buffer-get-the-current-buffer}

```emacs-lisp
(current-buffer)
```


#### get-buffer: get a buffer by name {#get-buffer-get-a-buffer-by-name}

```emacs-lisp
(get-buffer "*scratch*")
```


#### get-buffer-create: create the buffer if not exist {#get-buffer-create-create-the-buffer-if-not-exist}

```emacs-lisp
(get-buffer-create "yaya")
```


#### changing the current buffer {#changing-the-current-buffer}

```emacs-lisp
(progn
  (set-buffer (get-buffer "*scratch*"))
  (current-buffer))
```


#### Goto a buffer {#goto-a-buffer}

```emacs-lisp
(with-current-buffer "*BUF*"
  ;; do something like progn
  )
```


#### Changing the current buffer safely {#changing-the-current-buffer-safely}

It will return to the original buffer after the operation finished.

```emacs-lisp
(progn
  (save-current-buffer
    (set-buffer "*scratch*")
    (message "Current buffer: %s" (current-buffer)))
  (current-buffer))
```


#### Working with file buffers {#working-with-file-buffers}

To get the full file path for the file that the buffer represents

```emacs-lisp
(buffer-file-name)
```

To find a buffer that represents a particular file

```emacs-lisp
(get-file-buffer "/Users/yanchunwei/project/myblog2022/content-org/emacs-essentials.org")
```


#### Loading a file into a buffer without display it {#loading-a-file-into-a-buffer-without-display-it}

```emacs-lisp
(find-file-noselect "xx.org")
```


#### Get all buffer names {#get-all-buffer-names}

```emacs-lisp
(mapcar #'buffer-name (buffer-list))
```


### point {#point}

The "point" is the location of the cursor in the buffer.

```emacs-lisp
(point)
```

```emacs-lisp
(point-max)
```

```emacs-lisp
(point-min)
```


#### Moving the point {#moving-the-point}

```emacs-lisp
(goto-char 1)
(goto-char (point-max))

;; goto the begining of the buffer
(beginning-of-buffer)

;; goto the end of the buffer
(end-of-buffer)

(forward-char)
(forward-char 5)

(forward-word)
(backward-word)
```


#### Preserving the point {#preserving-the-point}

```emacs-lisp
(save-excursion
  (goto-char (point-max))
  (point)
  )
```


#### Examining buffer text {#examining-buffer-text}

To look at text in the buffer.

```emacs-lisp
(char-after)
(char-after (point))
(char-after (point-min))
```


#### The Thing {#the-thing}

The `thing-at-point` function is very useful for grabbing the text at the point if it matches a particular type of "thing".

```emacs-lisp
(thing-at-point 'word)
```

```emacs-lisp
(thing-at-point 'sentence)
```

```emacs-lisp
(thing-at-point 'sentence t)
```


#### Serching for text {#serching-for-text}

```emacs-lisp
(search-forward "thing")
```


#### Inserting text {#inserting-text}

```emacs-lisp
(insert "000")
(insert "\n" "This is" ?\s ?\n "Sparta!")
```


#### Deleting text {#deleting-text}

```emacs-lisp
(with-current-buffer ".gitignore"
  (delete-region (point) (point-max)))
```


#### Saving a buffer {#saving-a-buffer}

To save the contents of a buffer back to the file it is associated with

```emacs-lisp
(save-buffer)
```


### org-element {#org-element}


### file and path {#file-and-path}


#### Get the path of the current file {#get-the-path-of-the-current-file}

The `buffer-file-name` is a buffer builtin variable holding the file name of the current buffer.

```emacs-lisp
(file-truename buffer-file-name)
```


#### Get path without suffix {#get-path-without-suffix}

```emacs-lisp
(file-name-sans-extension "/tmp/a.org")
```


#### Write to file {#write-to-file}

Overwrite the content:

```emacs-lisp
(with-temp-file "/tmp/1.org"
  (insert "hello world")
  (message "file content: %s" (buffer-string))
  )
```


### execute shell command {#execute-shell-command}

```emacs-lisp
(shell-command "echo hello")
```


## Modern libraries {#modern-libraries}


### ht.el for hashtables {#ht-dot-el-for-hashtables}

Reference [ht.el](https://github.com/Wilfred/ht.el) for more details.


#### creating a hash table {#creating-a-hash-table}

Create an empty hash table

```emacs-lisp
(let* ((the-dic (ht-create)))
  the-dic
  )
```

Create a hash table with initial records

```emacs-lisp
(let* ((the-dic (ht
                 ("name" "Tom")
                 ("sex" 'male))))
  the-dic
  )
```


#### accessing the hash table {#accessing-the-hash-table}

```emacs-lisp
(let* ((the-dic (ht ("name" "Tom") ("sex" 'male))))
  ;; get a record
  ;; returns "Tom"
  (ht-get the-dic "name")
  )
```


#### Iterating over the hash table {#iterating-over-the-hash-table}

Readonly mapping:

```emacs-lisp
(let* ((the-dic (ht ("name" "Tom") ("sex" 'male) ("age" 18))))
  (ht-map (lambda (key value) (message "%S: %S" key value)) the-dic)
  )
```

Mutable mapping:

```emacs-lisp
(let* ((the-dic (ht ("name" "Tom") ("sex" 'male) ("age" 18))))
  (ht-map (lambda (key value)
            ;; modify the value if is string
            (setf value (if (stringp value)
                            (concat "modified " value)
                          value))) the-dic))
```
