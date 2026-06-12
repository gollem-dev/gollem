package reflexion_test

import (
	"testing"

	"github.com/gollem-dev/gollem/strategy/reflexion"
	"github.com/m-mizutani/gt"
)

func TestMemory_AddAndGet(t *testing.T) {
	m := reflexion.NewMemory(3)

	// Initially empty
	gt.Equal(t, 0, m.Size())
	gt.Equal(t, 0, len(m.GetAll()))

	// Add first entry
	m.Add(1, "first reflection")
	gt.Equal(t, 1, m.Size())
	entries := m.GetAll()
	gt.Equal(t, 1, len(entries))
	gt.Equal(t, 1, entries[0].TrialNum)
	gt.Equal(t, "first reflection", entries[0].Reflection)

	// Add second entry
	m.Add(2, "second reflection")
	gt.Equal(t, 2, m.Size())
	entries = m.GetAll()
	gt.Equal(t, 2, len(entries))
	gt.Equal(t, 2, entries[1].TrialNum)
}

func TestMemory_FIFO(t *testing.T) {
	m := reflexion.NewMemory(3)

	// Fill memory to capacity
	m.Add(1, "reflection 1")
	m.Add(2, "reflection 2")
	m.Add(3, "reflection 3")
	gt.Equal(t, 3, m.Size())

	// Add fourth entry - should evict first
	m.Add(4, "reflection 4")
	gt.Equal(t, 3, m.Size())

	entries := m.GetAll()
	// First entry should be trial 2 now
	gt.Equal(t, 2, entries[0].TrialNum)
	gt.Equal(t, "reflection 2", entries[0].Reflection)
	// Last entry should be trial 4
	gt.Equal(t, 4, entries[2].TrialNum)
	gt.Equal(t, "reflection 4", entries[2].Reflection)
}

func TestMemory_SingleEntry(t *testing.T) {
	m := reflexion.NewMemory(1)

	m.Add(1, "first")
	gt.Equal(t, 1, m.Size())

	// Adding second should replace first
	m.Add(2, "second")
	gt.Equal(t, 1, m.Size())

	entries := m.GetAll()
	gt.Equal(t, 2, entries[0].TrialNum)
	gt.Equal(t, "second", entries[0].Reflection)
}
