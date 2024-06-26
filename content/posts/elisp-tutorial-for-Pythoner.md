+++
title = "Emacs Lisp Introduction for Python Programmers"
author = ["Chunwei Yan"]
date = 2024-04-28
tags = ["emacs", "lisp", "tech", "python"]
draft = false
+++

This is a brief introduction to Emacs Lisp for Python programmers, (although I am not an Elisp expert, and actually I am more familiar with Python than Elisp). Both languages have quite different syntaxes, it is interesting to see how can implement Python code with lisp code.

The content follows the strucutre from [Learn X in Y Minutes Where X is Python](https://learnxinyminutes.com/docs/python/), and we will touch all the topics.


## Primitive Datatypes and Operators {#primitive-datatypes-and-operators}


### Numbers {#numbers}


{{< rawhtml >}}
<table>
<tr>
<td>
<b>Python</b>
<pre><code class="python-html">
# Integer
1
# Float
3.14
# Math is what you would expect
1 + 1   # => 2
8 - 1   # => 7
10 * 2  # => 20
35 / 5  # => 7.0

# Integer division rounds down for both positive and negative numbers.
5 // 3       # => 1
-5 // 3      # => -2
5.0 // 3.0   # => 1.0  # works on floats too
-5.0 // 3.0  # => -2.0

# The result of division is always a float
10.0 / 3  # => 3.3333333333333335

# Modulo operation
7 % 3   # => 1
# i % j have the same sign as j, unlike C
-7 % 3  # => 2

# Exponentiation (x**y, x to the yth power)
2**3  # => 8

# Enforce precedence with parentheses
1 + 3 * 2    # => 7
(1 + 3) * 2  # => 8
</code></pre>
</td>

<td>

<b>Elisp</b>
<pre><code class="lisp-html">
;; Integer
1
;; Float
3.14
;; Math is what you would expect
(+ 1 1)   ; => 2
(- 8 1)   ; => 7
(* 10 2)  ; => 20
(/ 35 5)  ; => 7

;; Integer division rounds down for both positive and negative numbers.
(truncate (/ 5 3))       ; => 1
(truncate (/ -5 3))      ; => -2
(truncate (/ 5.0 3.0))   ; => 1.0  ; works on floats too
(truncate (/ -5.0 3.0))  ; => -2.0

;; The result of division is always a float if the denominator or numerator is float
(/ 10.0 3)  ; => 3.3333333333333335

;; Modulo operation
(% 7 3)   ; => 1
;; different from Python
(% -7 3)  ; => -1

;; Exponentiation
(expt 2 3)  ; => 8

;; Enforce precedence with parentheses
(+ 1 (* 3 2))    ; => 7
(* (1+ 3) 2)  ; => 8

</code></pre>


</td>
</table>
{{< /rawhtml >}}


### Bools and comparasion {#bools-and-comparasion}

In Emacs Lisp, booleans are represented by the symbols `t` for true and `nil` for false.

{{< rawhtml >}}

<table>
<tr> <td>

<pre><code class="python-html">
# Boolean values are primitives (Note: the capitalization)
True   # => True
False  # => False

# negate with not
not True   # => False
not False  # => True

# Boolean Operators
# Note "and" and "or" are case-sensitive
True and False  # => False
False or True   # => True

# True and False are actually 1 and 0 but with different keywords
True + True  # => 2
True * 8     # => 8
False - 5    # => -5

# Comparison operators look at the numerical value of True and False
0 == False   # => True
2 > True     # => True
2 == True    # => False
-5 != False  # => True

# None, 0, and empty strings/lists/dicts/tuples/sets all evaluate to False.
# All other values are True
bool(0)      # => False
bool("")     # => False
bool([])     # => False
bool({})     # => False
bool(())     # => False
bool(set())  # => False
bool(4)      # => True
bool(-6)     # => True

bool(0)   # => False
bool(2)   # => True
0 and 2   # => 0
bool(-5)  # => True
bool(2)   # => True
-5 or 0   # => -5

# Equality is ==
1 == 1  # => True
2 == 1  # => False

# Inequality is !=
1 != 1  # => False
2 != 1  # => True

# More comparisons
1 < 10  # => True
1 > 10  # => False
2 <= 2  # => True
2 >= 2  # => True

# Seeing whether a value is in a range
1 < 2 and 2 < 3  # => True
2 < 3 and 3 < 2  # => False
# Chaining makes this look nicer
1 < 2 < 3  # => True
2 < 3 < 2  # => False

# (is vs. ==) is checks if two variables refer to the same object, but == checks
# if the objects pointed to have the same values.
a = [1, 2, 3, 4]  # Point a at a new list, [1, 2, 3, 4]
b = a             # Point b at what a is pointing to
b is a            # => True, a and b refer to the same object
b == a            # => True, a's and b's objects are equal
b = [1, 2, 3, 4]  # Point b at a new list, [1, 2, 3, 4]
b is a            # => False, a and b do not refer to the same object
b == a            # => True, a's and b's objects are equal
</code></pre>

</td>

<td>

<pre><code class="lisp-html">
;; Boolean values are symbols
t   ; => t
nil ; => nil

;; negate with not
(not t)   ; => nil
(not nil) ; => t

;; Boolean Operators
;; Use `and` and `or` for logical operations
(and t nil)  ; => nil
(or nil t)   ; => t

;; In Elisp, `t` or `nil` is not numeric, so numerical operations will fail
(+ t t)      ; error
(* t 8)      ; error
(- nil 5)    ; error

;; Comparison operators
(= 0 nil)    ; => error, since nil is not a numerical value
(eq 0 nil)   ; => nil, because `eq` is used for checking equality of objects
             ; similar to Python's `is`
(eq 1.0 1.0) ; => nil, both are different objects
(= 1.0 1.0)  ; => t, numerical equal
(eq 1 1)     ; => t, constant integers of same value share the same object

;; In Elisp, `nil` is the only false value. All other values are true.
(not 0)      ; => nil
(not "")     ; => nil
(not '())    ; => nil
(not (make-hash-table)) ; => nil
(not nil)    ; => t
(not 4)      ; => nil
(not -6)     ; => nil

(and nil 2)  ; => nil
(or -5 nil)  ; => -5

;; Equality is checked with `eq` for objects
;; and `equal` for value comparison, or `=` for numbers
(= 1 1)  ; => t
(= 2 1)  ; => nil

;; Inequality with `/=`
(/= 1 1)  ; => nil
(/= 2 1)  ; => t

;; More comparisons
(< 1 10)  ; => t
(> 1 10)  ; => nil
(<= 2 2)  ; => t
(>= 2 2)  ; => t

;; Logical combinations for range checking
;; Chaining like Python's `1 < 2 < 3` doesn't directly translate
(and (< 1 2) (< 2 3))  ; => t
(and (< 2 3) (< 3 2))  ; => nil

;; Setting up variables and lists
(setq a '(1 2 3 4))  ; Set 'a' to a new list
(setq b a)   ; Point 'b' at what 'a' is pointing to
(eq b a)     ; => t, 'a' and 'b' refer to the same object
(equal b a)  ; => t, 'a's and 'b's objects are equal

(setq b '(1 2 3 4))  ; Set 'b' to a new, but identical list
(eq b a)     ; => nil, 'a' and 'b' do not refer to the same object
(equal b a)  ; => t, 'a's and 'b's objects are equal

</code></pre>
</td> </tr>

</table>

{{< /rawhtml >}}

-   `eq` checks if two symbols or objects refer to the same memory address (identical objects).
-   `equal` tests for structural equality without considering if the two are the exact same object.


### String related {#string-related}

{{< rawhtml >}}

<table> <tr>

<td> <pre><code class="python-html">
# Strings are created with " or '
"This is a string."
'This is also a string.'

# Strings can be added too
"Hello " + "world!"  # => "Hello world!"
# String literals (but not variables) can be concatenated without using '+'
"Hello " "world!"    # => "Hello world!"

# A string can be treated like a list of characters
"Hello world!"[0]  # => 'H'

# You can find the length of a string
len("This is a string")  # => 16

# Since Python 3.6, you can use f-strings or formatted string literals.
name = "Reiko"
f"She said her name is {name}."  # => "She said her name is Reiko"
# Any valid Python expression inside these braces is returned to the string.
f"{name} is {len(name)} characters long."  # => "Reiko is 5 characters long."
</code></pre> </td>

<td> <pre><code class="lisp-html">

;; Strings are created with double quotes
"This is a string."
"This is also a string."

;; Strings can be concatenated using `concat`
(concat "Hello " "world!")  ; => "Hello world!"

;; In Elisp, there is no automatic concatenation without using a function like `concat`.

;; Accessing a character in a string
(aref "Hello world!" 0)  ; => 72  (returns the ASCII value of 'H')

;; You can convert the ASCII value to a character if needed
(char-to-string (aref "Hello world!" 0))  ; => "H"

;; Finding the length of a string
(length "This is a string")  ; => 16

;; Emacs Lisp doesn't have a built-in feature exactly like Python's f-strings,
;; but you can use `format` to achieve similar results.
(setq name "Reiko")
(format "She said her name is %s." name)  ; => "She said her name is Reiko."
;; Using `format` for more complex expressions
(format "%s is %d characters long." name (length name))  ; => "Reiko is 5 characters long."

</code></pre> </td>
</tr> </table>

{{< /rawhtml >}}


### Variables and Collections {#variables-and-collections}

{{< rawhtml >}}

<table> <tr>

<td> <pre><code class="language-python">

print("I'm Python. Nice to meet you!")  # => I'm Python. Nice to meet you!

# By default the print function also prints out a newline at the end.
# Use the optional argument end to change the end string.
print("Hello, World", end="!")  # => Hello, World!

# Simple way to get input data from console
input_string_var = input("Enter some data: ")  # Returns the data as a string

# There are no declarations, only assignments.
# Convention in naming variables is snake_case style
some_var = 5
some_var  # => 5

# Accessing a previously unassigned variable is an exception.
# See Control Flow to learn more about exception handling.
some_unknown_var  # Raises a NameError

# if can be used as an expression
# Equivalent of C's '?:' ternary operator
"yay!" if 0 > 1 else "nay!"  # => "nay!"

</code></pre> </td>

<td> <pre><code class="lisp-html">

;; Emacs Lisp has a print function, `message`, commonly used for displaying output in the echo area.
(message "I'm Emacs Lisp. Nice to meet you!")  ; Prints: I'm Emacs Lisp. Nice to meet you!

;; By default, `message` also outputs a newline at the end. You can concatenate strings to simulate different endings.
(message "Hello, World!")  ; Prints: Hello, World

;; Simple way to get input data from console
(setq input-string-var (read-string "Enter some data: "))  ; Prompts user and returns the data as a string

;; Variables are set with `setq`, and Emacs Lisp uses lower-case with dashes (lisp-case).
(setq some-var 5)
some-var  ; => 5

;; Accessing a previously unassigned variable results in `nil` if not set, not an exception.
some-unknown-var  ; => nil unless previously set, does not raise an error

;; `if` can be used similarly to the ternary operator
(if (> 0 1) "yay!" "nay!")  ; => "nay!"

</code></pre> </td>

</tr> </table>

{{< /rawhtml >}}


#### list {#list}

{{< rawhtml >}}

<table> <tr>

<td> <pre><code class="python-html">
# Lists store sequences
li = []
# You can start with a prefilled list
other_li = [4, 5, 6]

# Add stuff to the end of a list with append
li.append(1)    # li is now [1]
li.append(2)    # li is now [1, 2]
li.append(4)    # li is now [1, 2, 4]
li.append(3)    # li is now [1, 2, 4, 3]
# Remove from the end with pop
li.pop()        # => 3 and li is now [1, 2, 4]
# Let's put it back
li.append(3)    # li is now [1, 2, 4, 3] again.

# Access a list like you would any array
li[0]   # => 1
# Look at the last element
li[-1]  # => 3

# Looking out of bounds is an IndexError
li[4]  # Raises an IndexError

# Make a one layer deep copy using slices
li2 = li[:]  # => li2 = [1, 2, 4, 3] but (li2 is li) will result in false.

# Remove arbitrary elements from a list with "del"
del li[2]  # li is now [1, 2, 3]

# Remove first occurrence of a value
li.remove(2)  # li is now [1, 3]
li.remove(2)  # Raises a ValueError as 2 is not in the list

# Insert an element at a specific index
li.insert(1, 2)  # li is now [1, 2, 3] again

# Get the index of the first item found matching the argument
li.index(2)  # => 1
li.index(4)  # Raises a ValueError as 4 is not in the list

# You can add lists
# Note: values for li and for other_li are not modified.
li + other_li  # => [1, 2, 3, 4, 5, 6]

# Concatenate lists with "extend()"
li.extend(other_li)  # Now li is [1, 2, 3, 4, 5, 6]

# Check for existence in a list with "in"
1 in li  # => True

# Examine the length with "len()"
len(li)  # => 6

</code></pre> </td>

<td> <pre><code class="lisp-html">
;; Lists store sequences
(setq li '())
;; You can start with a prefilled list
(setq other-li '(4 5 6))
;; The '(...) is a macro of (list ...), so the below has same effect
(setq other-li (list 4 5 6))

;; Add stuff to the end of a list with `push` (note: `push` adds to the front, so let's use `append` for the end)
(setq li (append li '(1)))  ; li is now (1)
(setq li (append li '(2)))  ; li is now (1 2)
(setq li (append li '(4)))  ; li is now (1 2 4)
(setq li (append li '(3)))  ; li is now (1 2 4 3)
;; Remove from the end with `pop`
(pop li)  ; => 3 and li is now (1 2 4)
;; Let's put it back
(setq li (append li '(3)))  ; li is now (1 2 4 3) again.

;; Access a list like you would any array (using `nth`)
(nth 0 li)  ; => 1
;; Look at the last element (using `car` on the reversed list)
(car (last li))  ; => 3

;; Looking out of bounds does not raise an error by default, returns nil
(nth 4 li)  ; => nil, does not raise an IndexError

;; Make a one layer deep copy using `copy-sequence`
(setq li2 (copy-sequence li))  ;; li2 equals [1 2 4 3] but (eq li2 li) will result in nil.

;; Remove arbitrary elements from a list with `setf` and `nthcdr`
(setf (nthcdr 2 li) (cddr (nthcdr 2 li)))  ;; li is now [1 2 3]

;; Emacs Lisp does not have a direct equivalent of Python's `list.remove`
;; for non-destructive removal, you'd typically filter the list
(setq li (remove 2 li))  ;; li is now [1 3]
;; For handling error (when element is not in the list), Emacs Lisp usually uses `condition-case`
(condition-case nil
    (setq li (remove 2 li))
  (error (message "ValueError: 2 is not in the list")))

;; Insert an element at a specific index
(setq li (cl-list* 1 2 (cdr li)))  ;; li is now [1 2 3] again, using `cl-list*` to splice

;; Get the index of the first item found matching the argument
(position 2 li)  ;; => 1
(condition-case nil
    (position 4 li)
  (error (message "ValueError: 4 is not in the list")))

;; You can add lists
(setq result (append li other_li))  ;; => [1 2 3 4 5 6]

;; Concatenate lists with `append` (destructively with `nconc`)
(setq li (append li other_li))  ;; Now li is [1 2 3 4 5 6]

;; Check for existence in a list with `member`
(member 1 li)  ;; => (1 2 3 4 5 6) which is true-ish in Lisp (non-nil means true)

;; Examine the length with `length`
(length li)  ;; => 6

</code></pre> </td>

</tr> </table>

{{< /rawhtml >}}


#### Dict {#dict}

{{< rawhtml >}}

<table> <tr>

<td> <pre><code class="python-html">
# Dictionaries store mappings from keys to values
empty_dict = {}
# Here is a prefilled dictionary
filled_dict = {"one": 1, "two": 2, "three": 3}

# Note keys for dictionaries have to be immutable types. This is to ensure that
# the key can be converted to a constant hash value for quick look-ups.
# Immutable types include ints, floats, strings, tuples.
invalid_dict = {[1,2,3]: "123"}  # => Yield a TypeError: unhashable type: 'list'
valid_dict = {(1,2,3):[1,2,3]}   # Values can be of any type, however.

# Look up values with []
filled_dict["one"]  # => 1

# Get all keys as an iterable with "keys()". We need to wrap the call in list()
# to turn it into a list. We'll talk about those later.  Note - for Python
# versions <3.7, dictionary key ordering is not guaranteed. Your results might
# not match the example below exactly. However, as of Python 3.7, dictionary
# items maintain the order at which they are inserted into the dictionary.
list(filled_dict.keys())  # => ["three", "two", "one"] in Python <3.7
list(filled_dict.keys())  # => ["one", "two", "three"] in Python 3.7+


# Get all values as an iterable with "values()". Once again we need to wrap it
# in list() to get it out of the iterable. Note - Same as above regarding key
# ordering.
list(filled_dict.values())  # => [3, 2, 1]  in Python <3.7
list(filled_dict.values())  # => [1, 2, 3] in Python 3.7+

# Check for existence of keys in a dictionary with "in"
"one" in filled_dict  # => True
1 in filled_dict      # => False

# Looking up a non-existing key is a KeyError
filled_dict["four"]  # KeyError

# Use "get()" method to avoid the KeyError
filled_dict.get("one")      # => 1
filled_dict.get("four")     # => None
# The get method supports a default argument when the value is missing
filled_dict.get("one", 4)   # => 1
filled_dict.get("four", 4)  # => 4

# "setdefault()" inserts into a dictionary only if the given key isn't present
filled_dict.setdefault("five", 5)  # filled_dict["five"] is set to 5
filled_dict.setdefault("five", 6)  # filled_dict["five"] is still 5

# Adding to a dictionary
filled_dict.update({"four":4})  # => {"one": 1, "two": 2, "three": 3, "four": 4}
filled_dict["four"] = 4         # another way to add to dict

# Remove keys from a dictionary with del
del filled_dict["one"]  # Removes the key "one" from filled dict

# From Python 3.5 you can also use the additional unpacking options
{"a": 1, **{"b": 2}}  # => {'a': 1, 'b': 2}
{"a": 1, **{"a": 2}}  # => {'a': 2}
</code></pre> </td>

<td> <pre><code class="lisp-html">
;; Hash tables store mappings from keys to values
(setq empty-dict (make-hash-table))
;; Here is a prefilled hash table
(setq filled-dict (make-hash-table :test 'equal))
(puthash "one" 1 filled-dict)
(puthash "two" 2 filled-dict)
(puthash "three" 3 filled-dict)

;; Note keys for hash tables should be comparable with the test function, `equal` here allows strings
;; Emacs Lisp hash tables do not restrict key types as strictly as Python does by default.
;; Attempt to use mutable types such as lists can be handled but requires careful consideration of equality testing.

;; Look up values with `gethash`
(gethash "one" filled-dict)  ;; => 1

;; Get all keys as a list
(hash-table-keys filled-dict)  ;; => '("one" "two" "three") in Emacs Lisp, ordering depends on hash function

;; Get all values as a list
(hash-table-values filled-dict)  ;; => '(1 2 3)

;; Check for existence of keys in a hash table with `gethash`
(when (gethash "one" filled-dict) t)  ;; => t (true)
(when (gethash 1 filled-dict) t)      ;; => nil (false)

;; Looking up a non-existing key returns nil by default, no error
(gethash "four" filled-dict)  ;; => nil

;; Use `gethash` with a default value to avoid nil for non-existing keys
(gethash "one" filled-dict 4)   ;; => 1
(gethash "four" filled-dict 4)  ;; => 4

;; `sethash` inserts into a hash table, replacing any existing value for the key
(puthash "five" 5 filled-dict)  ;; filled-dict now has key "five" set to 5
(puthash "five" 6 filled-dict)  ;; filled-dict["five"] is updated to 6

;; Adding to a hash table
(puthash "four" 4 filled-dict)  ;; filled-dict now includes "four" => 4

;; Remove keys from a hash table with `remhash`
(remhash "one" filled-dict)  ;; Removes the key "one" from filled-dict

;; Unpacking and merging hash tables isn't a direct feature in Emacs Lisp,
;; but can be achieved through looping and setting keys.

;; Below is an example of how to "merge" two hash tables in Emacs Lisp.
(setq a (make-hash-table :test 'equal))
(setq b (make-hash-table :test 'equal))
(puthash "a" 1 a)
(puthash "b" 2 b)

;; Simulating Python's dict unpacking:
(maphash (lambda (k v) (puthash k v a)) b)
;; Now, 'a' contains the contents of both 'a' and 'b'

</code></pre> </td>
</tr> </table>

{{< /rawhtml >}}


#### Set {#set}

{{< rawhtml >}}

<table> <tr>

<td> <pre><code class="python-html">
# Sets store ... well sets
empty_set = set()
# Initialize a set with a bunch of values.
some_set = {1, 1, 2, 2, 3, 4}  # some_set is now {1, 2, 3, 4}

# Similar to keys of a dictionary, elements of a set have to be immutable.
invalid_set = {[1], 1}  # => Raises a TypeError: unhashable type: 'list'
valid_set = {(1,), 1}

# Add one more item to the set
filled_set = some_set
filled_set.add(5)  # filled_set is now {1, 2, 3, 4, 5}
# Sets do not have duplicate elements
filled_set.add(5)  # it remains as before {1, 2, 3, 4, 5}

# Do set intersection with &
other_set = {3, 4, 5, 6}
filled_set & other_set  # => {3, 4, 5}

# Do set union with |
filled_set | other_set  # => {1, 2, 3, 4, 5, 6}

# Do set difference with -
{1, 2, 3, 4} - {2, 3, 5}  # => {1, 4}

# Do set symmetric difference with ^
{1, 2, 3, 4} ^ {2, 3, 5}  # => {1, 4, 5}

# Check if set on the left is a superset of set on the right
{1, 2} >= {1, 2, 3}  # => False

# Check if set on the left is a subset of set on the right
{1, 2} <= {1, 2, 3}  # => True

# Check for existence in a set with in
2 in filled_set   # => True
10 in filled_set  # => False

# Make a one layer deep copy
filled_set = some_set.copy()  # filled_set is {1, 2, 3, 4, 5}
filled_set is some_set        # => False
</code></pre> </td>

<td> <pre><code class="lisp-html">
;; Sets store ... well, something akin to sets using hash tables
(setq empty-set (make-hash-table :test 'equal))

;; Initialize a "set" with a bunch of values
(setq some-set (make-hash-table :test 'equal))
(mapc (lambda (x) (puthash x t some-set)) '(1 1 2 2 3 4))  ;; some-set is now effectively {1, 2, 3, 4}

;; Similar to keys of a dictionary, elements of a "set" have to be comparable with the test function.
;; Invalid "set" construction would cause errors if attempted with non-hashable types.
;; This is an invalid line in Emacs Lisp and commented out:
;; (setq invalid-set (make-hash-table :test 'equal))
;; (puthash [1] t invalid-set)  ;; Would raise an error in a hypothetical correct context

(setq valid-set (make-hash-table :test 'equal))
(puthash (list 1) t valid-set)
(puthash 1 t valid-set)

;; Add one more item to the "set"
(setq filled-set some-set)
(puthash 5 t filled-set)  ;; filled-set is now effectively {1, 2, 3, 4, 5}
(puthash 5 t filled-set)  ;; it remains as before {1, 2, 3, 4, 5}

;; Set operations using hash tables require custom functions or cl-lib utilities:
;; Intersection (set1 & set2)
(setq other-set (make-hash-table :test 'equal))
(mapc (lambda (x) (puthash x t other-set)) '(3 4 5 6))
(setq intersection-set (cl-intersection (hash-table-keys filled-set) (hash-table-keys other-set) :test 'equal))

;; Union (set1 | set2)
(setq union-set (cl-union (hash-table-keys filled-set) (hash-table-keys other-set) :test 'equal))

;; Difference (set1 - set2)
(setq difference-set (cl-set-difference (hash-table-keys filled-set) (hash-table-keys other-set) :test 'equal))

;; Symmetric Difference (set1 ^ set2)
(setq symmetric-difference-set (cl-set-exclusive-or (hash-table-keys filled-set) (hash-table-keys other-set) :test 'equal))

;; Superset check
(cl-subsetp (hash-table-keys other-set) (hash-table-keys filled-set) :test 'equal)  ;; => nil (false, filled-set is not a superset)

;; Subset check
(cl-subsetp (hash-table-keys '(1 2)) (hash-table-keys '(1 2 3)) :test 'equal)  ;; => t (true, {1, 2} is a subset of {1, 2, 3})

;; Check for existence in a "set" with `gethash`
(gethash 2 filled-set)  ;; => t (true)
(gethash 10 filled-set) ;; => nil (false)

;; Make a one layer deep copy
(setq filled-set-copy (make-hash-table :test 'equal))
(maphash (lambda (k v) (puthash k v filled-set-copy)) filled-set)
(eq filled-set filled-set-copy)  ;; => nil

</code></pre> </td>

</tr> </table>

{{< /rawhtml >}}


## Control Flow and Iterables {#control-flow-and-iterables}


### if {#if}

{{< rawhtml >}}
<table> <tr>
<td> <pre><code class="python-html">
# Let's just make a variable
some_var = 5

# Here is an if statement. Indentation is significant in Python!
# Convention is to use four spaces, not tabs.
# This prints "some_var is smaller than 10"
if some_var > 10:
    print("some_var is totally bigger than 10.")
elif some_var < 10:    # This elif clause is optional.
    print("some_var is smaller than 10.")
else:                  # This is optional too.
    print("some_var is indeed 10.")

</code></pre> </td>
{{</rawhtml >}}

{{< rawhtml >}}
<td> <pre><code class="lisp-html">
;; Let's just make a variable
(setq some-var 5)

;; Here is an if statement
;; This prints "some_var is smaller than 10"
(if (> some-var 10)
    (message "some_var is totally bigger than 10.")
  (if (< some-var 10)    ;; This is like the elif in Python
      (message "some_var is smaller than 10.")
    (message "some_var is indeed 10.")))  ;; This is the else part

</code></pre> </td>
</tr> </table>
{{< /rawhtml >}}

{{< rawhtml >}}
<table> <tr>
<td> <pre><code class="python-html">
"""
For loops iterate over lists
prints:
    dog is a mammal
    cat is a mammal
    mouse is a mammal
"""
for animal in ["dog", "cat", "mouse"]:
    # You can use format() to interpolate formatted strings
    print("{} is a mammal".format(animal))

"""
"range(number)" returns an iterable of numbers
from zero up to (but excluding) the given number
prints:
    0
    1
    2
    3
"""
for i in range(4):
    print(i)

"""
"range(lower, upper)" returns an iterable of numbers
from the lower number to the upper number
prints:
    4
    5
    6
    7
"""
for i in range(4, 8):
    print(i)

"""
"range(lower, upper, step)" returns an iterable of numbers
from the lower number to the upper number, while incrementing
by step. If step is not indicated, the default value is 1.
prints:
    4
    6
"""
for i in range(4, 8, 2):
    print(i)

"""
Loop over a list to retrieve both the index and the value of each list item:
    0 dog
    1 cat
    2 mouse
"""
animals = ["dog", "cat", "mouse"]
for i, value in enumerate(animals):
    print(i, value)

"""
While loops go until a condition is no longer met.
prints:
    0
    1
    2
    3
"""
</code></pre> </td>
{{</rawhtml >}}

{{< rawhtml >}}
<td> <pre><code class="lisp-html">
;; For loops iterate over lists
(dolist (animal '("dog" "cat" "mouse"))
  ;; You can use `format` to interpolate formatted strings
  (message "%s is a mammal" animal))

;; "dotimes" is used to iterate over a sequence of numbers
(dotimes (i 4)
  (message "%d" i))

;; To create a range from 4 to 7 (inclusive in Python, but we adjust for Lisp)
(dotimes (i 4)
  (message "%d" (+ i 4)))

;; Range with a step
(let ((start 4)
      (end 8)
      (step 2))
  (while (< start end)
    (message "%d" start)
    (setq start (+ start step))))

;; Loop over a list to retrieve both the index and the value of each list item
(let ((animals '("dog" "cat" "mouse"))
      (i 0))
  (dolist (value animals)
    (message "%d %s" i value)
    (setq i (1+ i))))

;; While loops go until a condition is no longer met.
(let ((i 0))
  (while (< i 4)
    (message "%d" i)
    (setq i (1+ i))))

</code></pre> </td>
</tr> </table>
{{< /rawhtml >}}


### While {#while}

{{< rawhtml >}}
<table> <tr>
<td> <pre><code class="python-html">
"""
While loops go until a condition is no longer met.
prints:
    0
    1

   2
    3
"""
x = 0
while x < 4:
    print(x)
    x += 1  # Shorthand for x = x + 1
</code></pre> </td>
{{< /rawhtml >}}

{{< rawhtml >}}
<td> <pre><code class="lisp-html">
;; For loops iterate over lists
(dolist (animal '("dog" "cat" "mouse"))
  ;; You can use `format` to interpolate formatted strings
  (message "%s is a mammal" animal))

;; "dotimes" is used to iterate over a sequence of numbers
(dotimes (i 4)
  (message "%d" i))

;; To create a range from 4 to 7 (inclusive in Python, but we adjust for Lisp)
(dotimes (i 4)
  (message "%d" (+ i 4)))

;; Range with a step
(let ((start 4)
      (end 8)
      (step 2))
  (while (< start end)
    (message "%d" start)
    (setq start (+ start step))))

;; Loop over a list to retrieve both the index and the value of each list item
(let ((animals '("dog" "cat" "mouse"))
      (i 0))
  (dolist (value animals)
    (message "%d %s" i value)
    (setq i (1+ i))))

;; While loops go until a condition is no longer met.
(let ((i 0))
  (while (< i 4)
    (message "%d" i)
    (setq i (1+ i))))

</code></pre> </td>
</tr> </table>
{{< /rawhtml >}}


### try ... catch {#try-dot-dot-dot-catch}

{{< rawhtml >}}
<table> <tr>
<td> <pre><code class="python-html">
# Handle exceptions with a try/except block
try:
    # Use "raise" to raise an error
    raise IndexError("This is an index error")
except IndexError as e:
    pass                 # Refrain from this, provide a recovery (next example).
except (TypeError, NameError):
    pass                 # Multiple exceptions can be processed jointly.
else:                    # Optional clause to the try/except block. Must follow
                         # all except blocks.
    print("All good!")   # Runs only if the code in try raises no exceptions
finally:                 # Execute under all circumstances
    print("We can clean up resources here")

</code></pre> </td>
{{< /rawhtml >}}

{{< rawhtml >}}
<td> <pre><code class="lisp-html">
;; Handle exceptions with a condition-case
(condition-case err
    ;; Use `error` to raise an error
    (error "This is an index error")
  ;; Each error type to be caught is specified in a separate clause.
  (error (message "Caught an error: %s" (error-message-string err))) ;; Handle specific errors
  (index-error nil)  ;; No action taken, similar to 'pass' in Python
  (type-error nil))  ;; Handle multiple specific errors jointly like TypeError

;; Emacs Lisp does not have a direct equivalent of Python's else and finally clauses.
;; To simulate 'finally', you just continue writing code after the condition-case
(message "We can clean up resources here")

;; If you need to run something only if no error was raised, you would have to manage
;; it with additional flags or control flow structures outside the condition-case.
(let ((no-error t))
  (condition-case nil
      (progn
        ;; Potentially error-throwing code here
        (error "Potential Error"))
    (error (setq no-error nil)))  ;; On error, set flag to nil
  (if no-error
      (message "All good!")))  ;; This runs only if no error was raised

;; Finally, code that runs regardless of error presence
(message "This always runs, simulating 'finally'")

</code></pre> </td>
</tr> </table>
{{< /rawhtml >}}


### with statement {#with-statement}

{{< rawhtml >}}
<table> <tr>
<td> <pre><code class="python-html">
# Instead of try/finally to cleanup resources you can use a with statement
with open("myfile.txt") as f:
    for line in f:
        print(line)

# Writing to a file
contents = {"aa": 12, "bb": 21}
with open("myfile1.txt", "w") as file:
    file.write(str(contents))        # writes a string to a file

import json
with open("myfile2.txt", "w") as file:
    file.write(json.dumps(contents))  # writes an object to a file

# Reading from a file
with open("myfile1.txt") as file:
    contents = file.read()           # reads a string from a file
print(contents)
# print: {"aa": 12, "bb": 21}

with open("myfile2.txt", "r") as file:
    contents = json.load(file)       # reads a json object from a file
print(contents)
# print: {"aa": 12, "bb": 21}
</code></pre> </td>
{{< /rawhtml >}}

{{< rawhtml >}}
<td> <pre><code class="lisp-html">
;; Reading from a file
(with-temp-buffer
  (insert-file-contents "myfile.txt")
  (goto-char (point-min))
  (while (not (eobp))
    (message "%s" (buffer-substring (line-beginning-position) (line-end-position)))
    (forward-line 1)))

;; Writing to a file
(let ((contents (format "%s" '((aa . 12) (bb . 21)))))
  (with-temp-file "myfile1.txt"
    (insert contents)))

;; Emacs Lisp doesn't have a built-in JSON parser in its default environment,
;; but assuming json.el or similar is available:
(require 'json)
(let ((contents (json-encode '((aa . 12) (bb . 21)))))
  (with-temp-file "myfile2.txt"
    (insert contents)))

;; Reading from a file as a string
(let ((contents ""))
  (with-temp-buffer
    (insert-file-contents "myfile1.txt")
    (setq contents (buffer-string)))
  (message "Contents of myfile1.txt: %s" contents))

;; Reading from a file as JSON
(let ((contents nil))
  (with-temp-buffer
    (insert-file-contents "myfile2.txt")
    (setq contents (json-read-from-string (buffer-string))))
  (message "Contents of myfile2.txt: %s" contents))

</code></pre> </td>
</tr> </table>
{{< /rawhtml >}}


## Functions {#functions}


### Define a function {#define-a-function}

{{< rawhtml >}}
<table> <tr>
<td> <pre><code class="python-html">
# Use "def" to create new functions
def add(x, y):
    print("x is {} and y is {}".format(x, y))
    return x + y  # Return values with a return statement

# Calling functions with parameters
add(5, 6)  # => prints out "x is 5 and y is 6" and returns 11

# Another way to call functions is with keyword arguments
add(y=6, x=5)  # Keyword arguments can arrive in any order.

# You can define functions that take a variable number of
# positional arguments
def varargs(*args):
    return args

varargs(1, 2, 3)  # => (1, 2, 3)

# You can define functions that take a variable number of
# keyword arguments, as well
def keyword_args(**kwargs):
    return kwargs

# Let's call it to see what happens
keyword_args(big="foot", loch="ness")  # => {"big": "foot", "loch": "ness"}


# You can do both at once, if you like
def all_the_args(*args, **kwargs):
    print(args)
    print(kwargs)
"""
all_the_args(1, 2, a=3, b=4) prints:
    (1, 2)
    {"a": 3, "b": 4}
"""

# When calling functions, you can do the opposite of args/kwargs!
# Use * to expand args (tuples) and use ** to expand kwargs (dictionaries).
args = (1, 2, 3, 4)
kwargs = {"a": 3, "b": 4}
all_the_args(*args)            # equivalent: all_the_args(1, 2, 3, 4)
all_the_args(**kwargs)         # equivalent: all_the_args(a=3, b=4)
all_the_args(*args, **kwargs)  # equivalent: all_the_args(1, 2, 3, 4, a=3, b=4)
</code></pre> </td>
{{< /rawhtml >}}

{{< rawhtml >}}
<td> <pre><code class="lisp-html">
;; Use `defun` to create new functions
(defun add (x y)
  (message "x is %d and y is %d" x y)
  (+ x y))  ;; Return values with an implicit return (last expression evaluated)

;; Calling functions with parameters
(add 5 6)  ;; => prints out "x is 5 and y is 6" and returns 11

;; Emacs Lisp does not support keyword arguments in the same way Python does,
;; but you can simulate them using a plist (property list).
(defun add-keywords (&rest args)
  (let ((x (plist-get args :x))
        (y (plist-get args :y)))
    (message "x is %d and y is %d" x y)
    (+ x y)))

(add-keywords :y 6 :x 5)  ;; Keyword arguments can arrive in any order, using plist.

;; You can define functions that take a variable number of
;; positional arguments
(defun varargs (&rest args)
  args)

(varargs 1 2 3)  ;; => (1 2 3)

;; You can define functions that take a variable number of
;; keyword arguments, as well
(defun keyword-args (&rest args)
  args)

;; Let's call it to see what happens
(keyword-args :big "foot" :loch "ness")  ;; => (:big "foot" :loch "ness")


;; You can do both at once, if you like
(defun all-the-args (&rest args)
  (message "args: %s" (prin1-to-string (cl-remove-if (lambda (x) (keywordp x)) args)))
  (message "kwargs: %s" (prin1-to-string (cl-loop for (key val) on args by #'cddr collect (cons key val)))))

;; all_the_args(1, 2, a=3, b=4) prints:
;;     args: (1 2)
;;     kwargs: ((:a . 3) (:b . 4))

;; When calling functions, you can use apply to expand args (lists)
(setq args '(1 2 3 4))
(setq kwargs '(:a 3 :b 4))
(apply 'all-the-args args)            ;; equivalent: all_the_args(1, 2, 3, 4)
(apply 'all-the-args (append args kwargs))  ;; equivalent: all_the_args(1, 2, 3, 4, :a 3, :b 4)

</code></pre> </td>
</tr> </table>
{{< /rawhtml >}}


### Global scopes {#global-scopes}

{{< rawhtml >}}
<table> <tr>
<td> <pre><code class="python-html">
# global scope
x = 5

def set_x(num):
    # local scope begins here
    # local var x not the same as global var x
    x = num    # => 43
    print(x)   # => 43

def set_global_x(num):
    # global indicates that particular var lives in the global scope
    global x
    print(x)   # => 5
    x = num    # global var x is now set to 6
    print(x)   # => 6

set_x(43)
set_global_x(6)
"""
prints:
    43
    5
    6
"""
</code></pre> </td>
{{< /rawhtml >}}

{{< rawhtml >}}
<td> <pre><code class="lisp-html">
;; Global scope
(defvar x 5)

;; `let` or `let*` is a safe way to get a local scope without altering
;; the global variables directly.

(defun set-x (num)
  ;; Local scope begins here
  (let ((x num))    ;; Local var x, not the same as global var x
    (message "%d" x)))  ;; => 43

(defun set-global-x (num)
  ;; This function uses the global x, if no `let` wrap a local scope
  (message "%d" x)   ;; => 5 (initial global value)
  (setq x num)       ;; Global var x is now set to 6
  (message "%d" x))  ;; => 6

(set-x 43)
(set-global-x 6)
;; This will print:
;; 43
;; 5
;; 6

</code></pre> </td>
</tr> </table>
{{< /rawhtml >}}


### Closures {#closures}

{{< rawhtml >}}
<table> <tr>
<td> <pre><code class="python-html">
# Python has first class functions
def create_adder(x):
    def adder(y):
        return x + y
    return adder

add_10 = create_adder(10)
add_10(3)   # => 13

# Closures in nested functions:
# We can use the nonlocal keyword to work with variables in nested scope which shouldn't be declared in the inner functions.
def create_avg():
    total = 0
    count = 0
    def avg(n):
        nonlocal total, count
        total += n
        count += 1
        return total/count
    return avg
avg = create_avg()
avg(3)  # => 3.0
avg(5)  # (3+5)/2 => 4.0
avg(7)  # (8+7)/3 => 5.0
</code></pre> </td>
{{< /rawhtml >}}

{{< rawhtml >}}
<td> <pre><code class="lisp-html">
;; Emacs Lisp has first-class functions
(defun create-adder (x)
  (lambda (y) (+ x y)))

(setq add-10 (create-adder 10))
(funcall add-10 3)   ;; => 13

;; Closures in nested functions
(defun create-avg ()
  (let ((total 0) (count 0))
    (lambda (n)
      (setq total (+ total n))
      (setq count (1+ count))
      (/ (float total) count))))

(setq avg (create-avg))
(funcall avg 3)  ;; => 3.0
(funcall avg 5)  ;; => 4.0
(funcall avg 7)  ;; => 5.0

</code></pre> </td>
</tr> </table>
{{< /rawhtml >}}


### lambda function {#lambda-function}

{{< rawhtml >}}
<table> <tr>
<td> <pre><code class="python-html">
# There are also anonymous functions
(lambda x: x > 2)(3)                  # => True
(lambda x, y: x ** 2 + y ** 2)(2, 1)  # => 5

# There are built-in higher order functions
list(map(add_10, [1, 2, 3]))          # => [11, 12, 13]
list(map(max, [1, 2, 3], [4, 2, 1]))  # => [4, 2, 3]

list(filter(lambda x: x > 5, [3, 4, 5, 6, 7]))  # => [6, 7]

</code></pre> </td>
{{< /rawhtml >}}

{{< rawhtml >}}
<td> <pre><code class="lisp-html">
;; Anonymous functions
(funcall (lambda (x) (> x 2)) 3)                  ;; => t (true in Emacs Lisp)
(funcall (lambda (x y) (+ (* x x) (* y y))) 2 1)  ;; => 5

;; First, let's assume 'add-10' is already defined as in the previous example
(setq add-10 (create-adder 10))

;; There are built-in higher order functions
(mapcar add-10 '(1 2 3))                        ;; => (11 12 13)
(mapcar #'max '(1 2 3) '(4 2 1))                ;; => (4 2 3)

;; Filter using a lambda
(remove-if-not (lambda (x) (> x 5)) '(3 4 5 6 7))  ;; => (6 7)

</code></pre> </td>
</tr> </table>
{{< /rawhtml >}}


## Modules {#modules}

{{< rawhtml >}}
<table> <tr>
<td> <pre><code class="python-html">
# You can import modules
import math
print(math.sqrt(16))  # => 4.0

# You can get specific functions from a module
from math import ceil, floor
print(ceil(3.7))   # => 4
print(floor(3.7))  # => 3

# You can import all functions from a module.
# Warning: this is not recommended
from math import *

# You can shorten module names
import math as m
math.sqrt(16) == m.sqrt(16)  # => True

# Python modules are just ordinary Python files. You
# can write your own, and import them. The name of the
# module is the same as the name of the file.

# You can find out which functions and attributes
# are defined in a module.
import math
dir(math)

# If you have a Python script named math.py in the same
# folder as your current script, the file math.py will
# be loaded instead of the built-in Python module.
# This happens because the local folder has priority
# over Python's built-in libraries.
</code></pre> </td>
{{< /rawhtml >}}

{{< rawhtml >}}
<td> <pre><code class="lisp-html">
;; You can require built-in "modules" (libraries in Emacs terms)
(require 'calc)  ;; Emacs's built-in calculator library, similar to importing 'math'
(message "%s" (calc-eval "sqrt(16)"))  ;; => "4.0"

;; In Emacs, you generally use `require` for modules and access functions directly.
;; There isn't a direct equivalent of Python's `from module import specific_function`,
;; but you access everything directly once the library is loaded.

;; Emacs doesn't support `import *` as Python does. Everything is accessible after `require`.

;; You can use `require` with a nickname, but it's less common than in Python.
;; More commonly, Emacs Lisp doesn't rename libraries; it accesses all exported symbols
;; directly after loading them.

;; Loading your own modules is similar to Python:
;; If you write your own Emacs Lisp file, say `my-module.el`, you can load it using:
(load "my-module")  ;; Equivalent to Python's import for custom modules.

;; You can list available functions and variables in a library using `C-h f` (for functions)
;; or `C-h v` (for variables) after loading the library, rather than using `dir()` like in Python.

;; Just as with Python, if you have an Emacs Lisp file in your load-path with the same name
;; as a built-in library, it will be loaded instead of the built-in one if you call `load`
;; explicitly with its filename.

;; Here's an example using `cl-lib` which is a common library for utility functions
(require 'cl-lib)
(message "%s" (cl-lib-version))  ;; Access a specific function or variable from `cl-lib`

;; Note: In practice, Emacs Lisp files (.el) when loaded or required, are typically
;; not referred to with an alias like Python's `as` but are loaded and

</code></pre> </td>
</tr> </table>
{{< /rawhtml >}}


## Class {#class}

{{< rawhtml >}}
<table> <tr>
<td> <pre><code class="python-html">
# We use the "class" statement to create a class
class Human:

    # A class attribute. It is shared by all instances of this class
    species = "H. sapiens"

    # Basic initializer, this is called when this class is instantiated.
    # Note that the double leading and trailing underscores denote objects
    # or attributes that are used by Python but that live in user-controlled
    # namespaces. Methods(or objects or attributes) like: __init__, __str__,
    # __repr__ etc. are called special methods (or sometimes called dunder
    # methods). You should not invent such names on your own.
    def __init__(self, name):
        # Assign the argument to the instance's name attribute
        self.name = name

        # Initialize property
        self._age = 0   # the leading underscore indicates the "age" property is
                        # intended to be used internally
                        # do not rely on this to be enforced: it's a hint to other devs

    # An instance method. All methods take "self" as the first argument
    def say(self, msg):
        print("{name}: {message}".format(name=self.name, message=msg))

    # Another instance method
    def sing(self):
        return "yo... yo... microphone check... one two... one two..."

    # A class method is shared among all instances
    # They are called with the calling class as the first argument
    @classmethod
    def get_species(cls):
        return cls.species

    # A static method is called without a class or instance reference
    @staticmethod
    def grunt():
        return "*grunt*"

    # A property is just like a getter.
    # It turns the method age() into a read-only attribute of the same name.
    # There's no need to write trivial getters and setters in Python, though.
    @property
    def age(self):
        return self._age

    # This allows the property to be set
    @age.setter
    def age(self, age):
        self._age = age

    # This allows the property to be deleted
    @age.deleter
    def age(self):
        del self._age
</code></pre> </td>
{{< /rawhtml >}}

{{< rawhtml >}}
<td> <pre><code class="lisp-html">

;; There is no built-in approach to define a struct.
;; The cl-lib extension brings `cl-defstruct` from Common Lisp.

;; Define a struct to mimic a class
(cl-defstruct (human
               (:constructor create-human (name))  ;; constructor function
               (:conc-name human-))  ;; prefix for automatically generated accessor functions
  name  ;; This will create human-name accessor
  (age 0)  ;; Default age, creating human-age accessor and mutator
  (species "H. sapiens"))  ;; A default class-like attribute, shared unless overridden

;; Instance method equivalent
(defun human-say (this msg)
  (message "%s: %s" (human-name this) msg))

(defun human-sing ()
  "yo... yo... microphone check... one two... one two...")

;; Class method equivalent
(defun human-get-species (this)
  (human-species this))

;; Static method equivalent
(defun human-grunt ()
  "*grunt*")

;; Using the struct with methods
(let ((bob (create-human "Bob")))
  (human-say bob "Hello!")  ;; Bob: Hello!
  (message "Bob sings: %s" (human-sing))  ;; Bob sings: yo... yo... microphone check... one two... one two...
  (message "Species: %s" (human-get-species bob))  ;; Species: H. sapiens
  (setf (human-age bob) 25)  ;; Setting age
  (message "Bob's age: %d" (human-age bob))  ;; Bob's age: 25
  (message "Static call: %s" (human-grunt)))  ;; Static call: *grunt*

;; Properties as getters/setters are handled by the `cl-defstruct` accessors and mutators
;; `age` property management is already provided by the struct definition

</code></pre> </td>
</tr> </table>
{{< /rawhtml >}}


## References {#references}

-   [Learn Python in Y Minutes](https://learnxinyminutes.com/docs/python/)
-   ChatGPT helps to generate most of the code examples

{{< rawhtml >}}
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/default.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>

<!-- and it's easy to individually load additional languages -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/go.min.js"></script>

<script>hljs.highlightAll();</script>
{{< / rawhtml >}}


{{< rawhtml >}}

<style>
table td {
display: block;
float: left;
width: 50%;
}
</style>

{{< /rawhtml >}}
