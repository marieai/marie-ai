// W1.0 feasibility spike (GATE).
// Goal: prove wasmtime-go/v33 can instantiate the `node` WASI-P2 component and
// call execute(...) with a working http-client host import.
//
// Finding: wasmtime-go/v33 exposes ONLY the core-module API (NewModule /
// NewModuleFromFile / NewInstance / NewLinker). There is no Component type or
// component linker in the package. The checked-in fixture is a WASI-P2
// *component* (header `00 61 73 6d 0d 00 01 00`, layer=0x0001), not a core
// module (`... 01 00 00 00`). This program demonstrates that loading the
// component through the only available entry point fails.
package main

import (
	"fmt"
	"os"

	"github.com/bytecodealliance/wasmtime-go/v33"
)

func main() {
	const fixture = "../../../packages/marie-wasm/nodes/compiled/http-request.wasm"

	engine := wasmtime.NewEngine()

	// The ONLY load entry points in wasmtime-go/v33 are core-module loaders.
	// There is no NewComponentFromFile / NewComponent / component linker.
	_, err := wasmtime.NewModuleFromFile(engine, fixture)
	if err != nil {
		fmt.Printf("SPIKE RESULT: wasmtime-go/v33 cannot load the fixture as a core module: %v\n", err)
		fmt.Println("CONCLUSION: no Component Model support in wasmtime-go/v33 -> W1.0 GATE FAIL (in-process wasmtime-go).")
		os.Exit(0)
	}
	fmt.Println("UNEXPECTED: fixture loaded as a core module; revisit the gate analysis.")
}
