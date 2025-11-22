# Contributors' Guide

Thanks for your interest in contributing to BiGSTARS.jl!

Contributions of all kinds are welcome: bug reports, documentation improvements, 
new examples, performance tweaks, and core features. 
The best way to get started is to open a GitHub [issue](https://github.com/subhk/BiGSTARS.jl/issues) 

We follow the [ColPrac guide](https://github.com/SciML/ColPrac) for collaborative practices. 
New contributors are encouraged to read it before making their first contribution.

## 1. Set up a development environment

We assume you work from your own fork of the repository.

### Step 1: Fork & clone

On GitHub, fork:

``https://github.com/subhk/BiGSTARS.jl``

Then on your machine:

```bash
git clone https://github.com/<your-username>/BiGSTARS.jl.git
cd BiGSTARS.jl
```

### Step 2: Instantiate the project

From the repository root:

```julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## 2. Run the tests

After the environment is instantiated:

**From the shell:**

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```


## 3. Build the documentation

The documentation has its own environment in ``docs/Project.toml``

### Step 1: Instantiate the docs environment
**From the shell:**

```julia
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
```

### Step 2: Build the docs

```julia
julia --project=docs -e 'include("docs/make.jl")'
```

The HTML files will be written to ``docs/build/``.
Open ``docs/build/index.html`` in a browser to preview.

If you change the public API or examples, please update the docs and make sure they build without errors.


## 4. Pull request checklist
Before opening a PR:

- [] You created a feature/topic branch (not working on main).
- [] Pkg.test() passes.
- [] Docs are updated if behavior or API changed.
- [] The PR description briefly explains:
  - What you changed.
  - Why it’s useful.
  - How you tested it.

Thank you for helping improve BiGSTARS.jl!

