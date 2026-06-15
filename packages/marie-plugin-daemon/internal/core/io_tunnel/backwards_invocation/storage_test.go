package backwards_invocation

import (
	"bytes"
	"errors"
	"fmt"
	"sync"
	"testing"
)

const testTenant = "org1__ws1"

func TestStorageSetGetDeleteRoundTrip(t *testing.T) {
	store := NewStorage(t.TempDir())

	if err := store.Set(testTenant, "ext.test.kv", "checkpoint", []byte("v1")); err != nil {
		t.Fatalf("set failed: %v", err)
	}
	value, ok, err := store.Get(testTenant, "ext.test.kv", "checkpoint")
	if err != nil || !ok || !bytes.Equal(value, []byte("v1")) {
		t.Fatalf("get returned %q ok=%v err=%v", value, ok, err)
	}

	if err := store.Set(testTenant, "ext.test.kv", "checkpoint", []byte("v2")); err != nil {
		t.Fatalf("overwrite failed: %v", err)
	}
	value, ok, err = store.Get(testTenant, "ext.test.kv", "checkpoint")
	if err != nil || !ok || !bytes.Equal(value, []byte("v2")) {
		t.Fatalf("get after overwrite returned %q ok=%v err=%v", value, ok, err)
	}

	if err := store.Delete(testTenant, "ext.test.kv", "checkpoint"); err != nil {
		t.Fatalf("delete failed: %v", err)
	}
	_, ok, err = store.Get(testTenant, "ext.test.kv", "checkpoint")
	if err != nil || ok {
		t.Fatalf("expected missing after delete, got ok=%v err=%v", ok, err)
	}

	// Deleting a missing key is a no-op.
	if err := store.Delete(testTenant, "ext.test.kv", "checkpoint"); err != nil {
		t.Fatalf("delete of missing key failed: %v", err)
	}
}

func TestStorageGetMissing(t *testing.T) {
	store := NewStorage(t.TempDir())

	value, ok, err := store.Get(testTenant, "ext.test.kv", "never-set")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}
	if ok || value != nil {
		t.Fatalf("expected ok=false for missing key, got %q ok=%v", value, ok)
	}
}

func TestStorageValueCap(t *testing.T) {
	store := NewStorage(t.TempDir())

	if err := store.Set(testTenant, "ext.test.kv", "big", make([]byte, maxStorageValueBytes)); err != nil {
		t.Fatalf("set at cap failed: %v", err)
	}
	err := store.Set(testTenant, "ext.test.kv", "big", make([]byte, maxStorageValueBytes+1))
	if !errors.Is(err, ErrStorageValueTooBig) {
		t.Fatalf("expected storage_value_too_large, got %v", err)
	}
}

func TestStorageRejectsInvalidNames(t *testing.T) {
	store := NewStorage(t.TempDir())

	for _, names := range [][2]string{
		{"../escape", "ext.test.kv"},
		{"", "ext.test.kv"},
		{testTenant, "ext/test"},
		{testTenant, ".."},
	} {
		if err := store.Set(names[0], names[1], "k", []byte("v")); !errors.Is(err, ErrStorageInvalidName) {
			t.Fatalf("Set(%q, %q) expected storage_invalid_name, got %v", names[0], names[1], err)
		}
		if _, _, err := store.Get(names[0], names[1], "k"); !errors.Is(err, ErrStorageInvalidName) {
			t.Fatalf("Get(%q, %q) expected storage_invalid_name, got %v", names[0], names[1], err)
		}
		if err := store.Delete(names[0], names[1], "k"); !errors.Is(err, ErrStorageInvalidName) {
			t.Fatalf("Delete(%q, %q) expected storage_invalid_name, got %v", names[0], names[1], err)
		}
	}
}

func TestStorageConcurrentAccess(t *testing.T) {
	store := NewStorage(t.TempDir())

	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			key := fmt.Sprintf("key-%d", worker%2)
			for iteration := 0; iteration < 50; iteration++ {
				value := []byte(fmt.Sprintf("w%d-i%d", worker, iteration))
				if err := store.Set(testTenant, "ext.test.kv", key, value); err != nil {
					t.Errorf("set failed: %v", err)
					return
				}
				if _, _, err := store.Get(testTenant, "ext.test.kv", key); err != nil {
					t.Errorf("get failed: %v", err)
					return
				}
				if iteration%10 == 9 {
					if err := store.Delete(testTenant, "ext.test.kv", key); err != nil {
						t.Errorf("delete failed: %v", err)
						return
					}
				}
			}
		}(worker)
	}
	wg.Wait()
}

func TestStoragePersistsAcrossInstances(t *testing.T) {
	root := t.TempDir()

	if err := NewStorage(root).Set(testTenant, "ext.test.kv", "checkpoint", []byte("durable")); err != nil {
		t.Fatalf("set failed: %v", err)
	}
	value, ok, err := NewStorage(root).Get(testTenant, "ext.test.kv", "checkpoint")
	if err != nil || !ok || !bytes.Equal(value, []byte("durable")) {
		t.Fatalf("expected durable value across instances, got %q ok=%v err=%v", value, ok, err)
	}
}
