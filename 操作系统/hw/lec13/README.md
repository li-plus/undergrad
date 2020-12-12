# Sync and Mutex

## Case 4: two threads

```
CONCEPT: A shared variable named turn is used to keep track of whose turn it is to enter the critical section.
INITIALIZATION:

	shared int turn;
	...
	turn = i ;
ENTRY PROTOCOL (for Process i ):
	/* wait until it's our turn */
	while (turn != i ) {
	}
EXIT PROTOCOL (for Process i ):
	/* pass the turn on */
	turn = j ;
```

Infeasible! Failure case:

```
// process i:
ENTRY PROTOCOL (for Process i ):
	/* wait until it's our turn */
	while (turn != i ) {
	}
EXIT PROTOCOL (for Process i ):
	/* pass the turn on */
	turn = j ;
------------------------------------------------
// process i:
ENTRY PROTOCOL (for Process i ):
	/* wait until it's our turn */
	while (turn != i ) {
	}
```

Process i keeps waiting even if no process is in the critical section.

## Case 5: two threads

```
CONCEPT: A shared Boolean array named flags contains a flag for each process. The flag values are BUSY when the process is in its critical section (using the resource), or FREE when it is not.
INITIALIZATION:

	typedef char boolean;
	...
	shared boolean flags[n - 1];
	...
	flags[i ] = FREE;
	...
	flags[j ] = FREE;
	...
ENTRY PROTOCOL (for Process i ):
	/* wait while the other process is in its CS */
	while (flags[j ] == BUSY) {
	}
-->
	/* claim the resource */
	flags[i ] = BUSY;
EXIT PROTOCOL (for Process i ):
	/* release the resource */
	flags[i ] = FREE;
```

Infeasible! Failure case:

```
// process i:
ENTRY PROTOCOL (for Process i ):
	/* wait while the other process is in its CS */
	while (flags[j ] == BUSY) {
	}
------------------------------------------------
// process j:
ENTRY PROTOCOL (for Process j ):
	/* wait while the other process is in its CS */
	while (flags[i ] == BUSY) {
	}
```

Process i and j both enter the critical section.

## Case 6: two threads

```
CONCEPT: Again we use a shared Boolean array as in Algorithm 2. Each process sets its flag before  testing the other flag, thus avoiding the problem of violating mutual exclusion.
INITIALIZATION:

	typedef char boolean;
	...
	shared boolean flags[n -1];
	...
	flags[i ] = FREE;
	...
	flags[j ] = FREE;
	...
ENTRY PROTOCOL (for Process i ):
	/* claim the resource */
	flags[i ] = BUSY;
-->
	/* wait if the other process is using the resource */
	while (flags[j ] == BUSY) {
	}
EXIT PROTOCOL (for Process i ):
	/* release the resource */
	flags[i ] = FREE;
```

Infeasible! Failure case:

```
// process i
ENTRY PROTOCOL (for Process i ):
	/* claim the resource */
	flags[i ] = BUSY;
------------------------------------------------
// process j
ENTRY PROTOCOL (for Process j ):
	/* claim the resource */
	flags[j ] = BUSY;
------------------------------------------------
// process i
	while (flags[j ] == BUSY) {
	}
------------------------------------------------
// process j
	while (flags[i ] == BUSY) {
	}
```

Both flags are busy, causing dead lock. No one could enter the critical section.

## Case 7: two threads

```
CONCEPT: To avoid the deadlock problem of Algorithm 3, we periodically clear and reset our own flag while waiting for the other one.
INITIALIZATION:

	typedef char boolean;
	...
	shared boolean flags[n -1];
	...
	flags[i ] = FREE;
	...
	flags[j ] = FREE;
	...
ENTRY PROTOCOL (for Process i ):
	/* claim the resource */
	flags[i ] = BUSY;
-->
	/* wait if the other process is using the resource */
	while (flags[j ] == BUSY) {
		flags[i ] = FREE;
		delay a while ;
		flags[i ] = BUSY;
	}
EXIT PROTOCOL (for Process i ):
	/* release the resource */
	flags[i ] = FREE;
```

Infeasible! Failure case:

```
// process i
ENTRY PROTOCOL (for Process i ):
	/* claim the resource */
	flags[i ] = BUSY;
------------------------------------------------
// process j
ENTRY PROTOCOL (for Process j ):
	/* claim the resource */
	flags[j ] = BUSY;
------------------------------------------------
// process i
	while (flags[j ] == BUSY) {
------------------------------------------------
// process j
	while (flags[i ] == BUSY) {
------------------------------------------------
// process i
		flags[i ] = FREE;
		delay a while ;
------------------------------------------------
// process j
		flags[j ] = FREE;
		delay a while ;
------------------------------------------------
// process i
		flags[i ] = BUSY;
------------------------------------------------
// process j
   		flags[j ] = BUSY;
------------------------------------------------
// process i
	while (flags[j ] == BUSY) {
------------------------------------------------
// process j
	while (flags[i ] == BUSY) {
```

Both are stuck in while loop. Dead lock.

## Case 8: two threads

```
CONCEPT: Both the turn variable and the status flags are combined in a way which we (the requesting process) set our flag and then check our neighbor's flag. 

INITIALIZATION:

	typedef char boolean;
	...
	shared boolean flags[n -1];
	shared int turn;
	...
	turn = i ;
	...
	flags[i ] = FREE;
	...
	flags[j ] = FREE;
	...
ENTRY PROTOCOL (for Process i ):
	/* claim the resource */
	flags[i ] = BUSY;

	/* wait if the other process is using the resource */
	while (flags[j ] == BUSY) {

		/* if waiting for the resource, also wait our turn */
		if (turn != i ) {
		
			/* but release the resource while waiting */
			flags[i ] = FREE;
			while (turn != i ) {
			}
			flags[i ] = BUSY;
		}

	}
EXIT PROTOCOL (for Process i ):
	/* pass the turn on, and release the resource */
	turn = j ;
	flags[i ] = FREE;
```

This is feasible!

## Case 9: two threads

```
CONCEPT: Both the turn variable and the status flags are used.

INITIALIZATION:

	typedef char boolean;
	...
	shared boolean flags[n -1];
	shared int turn;
	...
	turn = i ;
	...
	flags[i ] = FREE;
	...
	flags[j ] = FREE;
	...
ENTRY PROTOCOL (for Process i ):
	/* claim the resource */
	flags[i ] = BUSY;

	/* give away the turn */
	turn = j ;
	/* wait while the other process is using the resource *and* has the turn */
	while ((flags[j ] == BUSY) && (turn != i )) {
	}
EXIT PROTOCOL (for Process i ):
	/* release the resource */
	flags[i ] = FREE;
```

This is Peterson algorithm. Feasible!
