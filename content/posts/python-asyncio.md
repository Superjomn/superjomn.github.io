+++
title = "Asyncio By Example: Implementing the Producer-Consumer Pattern"
author = ["Chunwei Yan"]
date = 2024-07-09
tags = ["python", "coroutine", "tech"]
draft = false
+++

## The Most Basic Case {#the-most-basic-case}

With corountines, we can define a produer and consumer without any need for threads. This simplifies our code and makes it more efficient.

```python
import asyncio

async def producer():
    for i in range(6):
        await asyncio.sleep(0.2)
        yield i

async def consumer():
    async for i in producer():
        print(i)

async def main():
    await asyncio.gather(consumer())

asyncio.run(main())
```

```text
0
1
2
3
4
5
```


## Work with Heavy IO {#work-with-heavy-io}

When working with heavy IO operations, we need to be careful not to block the event loop. Running heavy IO operations can block the current event loop, which would slow down the scheduling of all coroutines.

We can use `asyncio.to_thread` to run the operation in another thread, thus avoiding the schedule being affected.

```python
import asyncio
import time

def blocking_io():
    time.sleep(0.1)
    print('Blocking IO complete')

async def producer(queue):
    for i in range(4):
        request = await asyncio.to_thread(blocking_io)
        await queue.put(request)
        print('Produced 1 item')

async def consumer(queue):
    for i in range(4):
        item = await queue.get()
        print('Consumed 1 item')
        queue.task_done()

async def main():
    queue = asyncio.Queue()

    producer_task = asyncio.create_task(producer(queue))
    consumer_task = asyncio.create_task(consumer(queue))

    await asyncio.gather(producer_task, consumer_task)

asyncio.run(main())
```

```text
Blocking IO complete
Produced 1 item
Consumed 1 item
Blocking IO complete
Produced 1 item
Consumed 1 item
Blocking IO complete
Produced 1 item
Consumed 1 item
Blocking IO complete
Produced 1 item
Consumed 1 item
```


## Event driven {#event-driven}

When the producer is not a coutine, we can utilize `asyncio.Event` to connect it with coutine world.

```python
import asyncio
from asyncio import Queue, Event

# NOTE: add_request is a traditional function
def add_request(event:Event, queue:Queue, request:int):
    print(f"add_request: {request}")
    queue.put_nowait(request)
    event.set()

async def consumer(event:Event, queue:Queue):
    for i in range(6): # loop
        await event.wait()
        event.clear()
        request = queue.get_nowait()
        print(f"consume: {request}")

async def main():
    event = Event()
    queue = Queue()
    consumer_task = asyncio.create_task(consumer(event, queue))
    for i in range(6):
        add_request(event, queue, i)
        await asyncio.sleep(1)
    await consumer_task


asyncio.run(main())
```

```text
add_request: 0
consume: 0
add_request: 1
consume: 1
add_request: 2
consume: 2
add_request: 3
consume: 3
add_request: 4
consume: 4
add_request: 5
consume: 5
```


## Take away {#take-away}

Python coroutine can simplify the code for producer-consumer pattern, and reduce the necessary for threads or other inter-thread collaboration.

Use `asyncio.to_thread` to dispatch blocking IO to another thread and avoid slow down the current `event_loop`.

Use `asyncio.Event` and `asyncio.Queue` to connect real-world code with coroutines.
