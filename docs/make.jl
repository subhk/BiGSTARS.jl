using Documenter
using DocumenterCitations
using Literate
using CairoMakie
using Printf
using BiGSTARS
using StaticArrays
using SpecialFunctions

# -- Generate literated examples ----------------------------------------------
const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src", "literated")

mkpath(OUTPUT_DIR)

examples = ["Stone1971.jl", "rRBC.jl"]

@info "EXAMPLES_DIR: $EXAMPLES_DIR"
@info "OUTPUT_DIR: $OUTPUT_DIR"

for example in examples
    # Input and target output file paths
    input_file  = joinpath(EXAMPLES_DIR, example)
    output_file = joinpath(OUTPUT_DIR, "") #, replace(example, ".jl" => ".md"))
    @info "output_file: $output_file"
    try
        # Generate a simple markdown file without Documenter-specific @example blocks
        Literate.markdown(
            input_file,
            output_file;
            documenter       = false,
            flavor            = DocumenterFlavor(),
            include_comments = true,
            include_code     = true,
            include_output   = false,
            execute          = false
        )
    catch e
        @error "Failed to literate $input_file" exception=(e, catch_backtrace())
        rethrow()
    end
end

# -- Auto-generate @autodocs for module APIs ----------------------------------
const MODULES_DIR = joinpath(@__DIR__, "src", "modules")
mkpath(MODULES_DIR)
for mod in ["Stone1971", "rRBC"]
    file = joinpath(MODULES_DIR, "$(mod).md")
    open(file, "w") do io
        println(io, "# $(mod) API\n")
        println(io, "```@autodocs")
        println(io, "Modules = [BiGSTARS.$(mod)]")
        println(io, "```")
    end
end

# -- Build site ---------------------------------------------------------------
format = Documenter.HTML(
    collapselevel  = 2,
    prettyurls     = get(ENV, "CI", nothing) == "true",
    size_threshold = 2^21,
    canonical      = "https://BiGSTARS.github.io/BiGSTARSDocumentation/stable/"
)

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "references.bib"),
    style = :authoryear
)

@printf("Building docs…\n")
makedocs(
    format    = format,
    authors   = "Subhajit Kar and contributors",
    sitename  = "BiGSTARS.jl",
    modules   = [BiGSTARS],
    plugins   = [bib],
    doctest   = false,
    clean     = true,
    checkdocs = :none,
    pages     = [
        "Home"                => "index.md",
        "Installation"        => "installation_instructions.md",
        "Examples"            => [
            "Stone1971"       => "literated/Stone1971.md",
            "rRBC"            => "literated/rRBC.md"
        ],
        "Modules"             => [
            "Stone1971 API"   => "modules/Stone1971.md",
            "rRBC API"        => "modules/rRBC.md"
        ],
        "Contributor's Guide" => "contributing.md",
        "References"          => "references.md"
    ]
)

# -- Cleanup temporary files --------------------------------------------------
for file in filter(x -> occursin(r"\\.jld2|\\.nc", x),
                   walkdir(@__DIR__) |> Iterators.flatten)
    rm(file; force=true)
end

# -- Deploy to GitHub Pages --------------------------------------------------
@info "Deploying to GitHub Pages"
# must set DOCUMENTER_KEY in CI for write access
deploydocs(
    repo         = "BiGSTARS/BiGSTARSDocumentation",
    branch       = "gh-pages",
    devbranch    = "main",
    forcepush    = true,
    push_preview = false,
    versions     = ["stable" => "v^", "dev" => "dev"]
)
```
