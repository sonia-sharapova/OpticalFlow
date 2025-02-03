package workstealing

import (
	"sync/atomic"
	"unsafe"
)

// Task: unit of work to be processed
type Task struct {
	Files      []string // Batch of DICOM files to process
	StartIndex int      // Starting index in the overall sequence
	OutputPath string   // Where to save results
}

// Node represents a node in the deque
type node struct {
	task *Task
	next *node
	prev *node
}

// Deque is a lock-free double-ended queue implementation
type Deque struct {
	head unsafe.Pointer // *node
	tail unsafe.Pointer // *node
	size int64
}

// NewDeque creates a new empty deque
func NewDeque() *Deque {
	return &Deque{}
}

// PushBottom adds a task to the bottom of the deque (used by owner thread)
func (d *Deque) PushBottom(task *Task) {
	newNode := &node{task: task}

	for {
		tail := (*node)(atomic.LoadPointer(&d.tail))
		head := (*node)(atomic.LoadPointer(&d.head))

		if tail == nil {
			// Empty deque
			if atomic.CompareAndSwapPointer(&d.head, unsafe.Pointer(head), unsafe.Pointer(newNode)) {
				atomic.StorePointer(&d.tail, unsafe.Pointer(newNode))
				atomic.AddInt64(&d.size, 1)
				return
			}
			continue
		}

		// Add to tail
		newNode.prev = tail
		if atomic.CompareAndSwapPointer(&d.tail, unsafe.Pointer(tail), unsafe.Pointer(newNode)) {
			tail.next = newNode
			atomic.AddInt64(&d.size, 1)
			return
		}
	}
}

// PopBottom removes and returns a task from the bottom of the deque
func (d *Deque) PopBottom() *Task {
	for {
		tail := (*node)(atomic.LoadPointer(&d.tail))
		if tail == nil {
			return nil
		}

		prev := tail.prev
		if prev == nil {
			// Last node
			if atomic.CompareAndSwapPointer(&d.head, unsafe.Pointer(tail), nil) {
				atomic.StorePointer(&d.tail, nil)
				atomic.AddInt64(&d.size, -1)
				return tail.task
			}
			continue
		}

		// Remove from tail
		if atomic.CompareAndSwapPointer(&d.tail, unsafe.Pointer(tail), unsafe.Pointer(prev)) {
			prev.next = nil
			atomic.AddInt64(&d.size, -1)
			return tail.task
		}
	}
}

// PopTop removes and returns a task from the top of the deque (used by stealing threads)
func (d *Deque) PopTop() *Task {
	for {
		head := (*node)(atomic.LoadPointer(&d.head))
		if head == nil {
			return nil
		}

		next := head.next
		if next == nil {
			// Last node
			if atomic.CompareAndSwapPointer(&d.tail, unsafe.Pointer(head), nil) {
				atomic.StorePointer(&d.head, nil)
				atomic.AddInt64(&d.size, -1)
				return head.task
			}
			continue
		}

		// Remove from head
		if atomic.CompareAndSwapPointer(&d.head, unsafe.Pointer(head), unsafe.Pointer(next)) {
			next.prev = nil
			atomic.AddInt64(&d.size, -1)
			return head.task
		}
	}
}

// Size returns the current size of the deque
func (d *Deque) Size() int {
	return int(atomic.LoadInt64(&d.size))
}
