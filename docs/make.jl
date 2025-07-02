using Documenter, DocumenterCitations, Literate

using CairoMakie
using Printf
using BiGSTARS

#####
##### Generate literated examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src", "literated")

mkpath(OUTPUT_DIR)

examples = [
    "Stone1971.jl",
    "rRBC.jl"
]

@info "EXAMPLES_DIR: $EXAMPLES_DIR"
@info "OUTPUT_DIR: $OUTPUT_DIR"

for example in examples
    input_file  = joinpath(EXAMPLES_DIR, example)
    output_file = joinpath(OUTPUT_DIR, replace(example, ".jl" => ".md"))
    try
        Literate.markdown(
            input_file,
            output_file;
            documenter      = true,
            include_comments = true,
            include_code     = true,
            include_output   = true
        )
    catch e
        @error "Failed to literate $input_file" exception=(e, catch_backtrace())
        rethrow()
    end
end

#####
##### Build and deploy docs
#####

format = Documenter.HTML(
    collapselevel  = 2,
    prettyurls     = get(ENV, "CI", nothing) == "true",
    size_threshold = 2^21,
    canonical      = "https://github.com/BiGSTARS/BiGSTARSDocumentation/stable/"
)

bib_filepath = joinpath(dirname(@__FILE__), "src", "references.bib")
bib          = CitationBibliography(bib_filepath, style=:authoryear)

@printf "making makedocs... \n"

try
    makedocs(
        authors    = ["Subhajit Kar", "Contributors"],
        sitename   = "BiGSTARS.jl",
        modules    = [BiGSTARS],
        plugins    = [bib],
        format     = format,
        doctest    = true,
        clean      = true,
        checkdocs  = :all,
        pages      = Any[
            "Home"           => "index.md",
            "Installation"   => "installation_instructions.md",
            "Examples"       => Any[
                "Stone1971" => "literated/Stone1971.md",
                "rRBC"       => "literated/rRBC.md"
            ],
            "Modules"        => Any[
                "Stone1971 API" => "modules/Stone1971.md",
                "rRBC API"      => "modules/rRBC.md"
            ],
            "Contributor's Guide" => "contributing.md",
            "References"      => "references.md"
        ]
    )
catch e
    @error "makedocs failed" exception=(e, catch_backtrace())
    rethrow()
end

@info "Clean up temporary .jld2 and .nc files created by doctests or literated examples..."

function recursive_find(directory, pattern)
    mapreduce(vcat, walkdir(directory)) do (root, dirs, files)
        joinpath.(root, filter(contains(pattern), files))
    end
end

files = []
for pattern in [r"\.jld2", r"\.nc"]
    append!(files, recursive_find(@__DIR__, pattern))
end

for file in files
    rm(file; force=true)
end

# deploydocs setup remains as-is...
