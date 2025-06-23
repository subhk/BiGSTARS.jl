using Documenter, DocumenterCitations, Literate

using CairoMakie

using BiGSTARS

#####
##### Generate literated examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")

examples = [
    "Stone_1971.jl",
    "rRBC.jl"
]

for example in examples
    input_file = joinpath(EXAMPLES_DIR, example)
    output_file = joinpath(OUTPUT_DIR, replace(example, ".jl" => ".md"))
    Literate.markdown(input_file, output_file; 
                      documenter=true, 
                      include_comments=true, 
                      include_code=true, 
                      include_output=true)
end

#####
##### Build and deploy docs
#####

format = Documenter.HTML(
    title = "BiGSTARS.jl",
    authors = "BiGSTARS developers",
    repo = ""
    )

bib_filepath = joinpath(dirname(@__FILE__), "src/references.bib")
bib = CitationBibliography(bib_filepath, style=:authoryear)


makedocs = Documenter.make_docs(
     authors = "Subhajit Kar, and contributors",
    sitename = "BiGSTARS.jl",
     modules = [BiGSTARS],
     plugins = [bib],
      format = format,
     doctest = true,
       clean = true,
   checkdocs = :all,
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Examples" => "examples.md",
        "Contributor's guide" => "contributing.md",
        "References" => "references.md"
    ]
)


@info "Clean up temporary .jld2 and .nc output created by doctests or literated examples..."

"""
    recursive_find(directory, pattern)

Return list of filepaths within `directory` that contains the `pattern::Regex`.
"""
recursive_find(directory, pattern) =
    mapreduce(vcat, walkdir(directory)) do (root, dirs, files)
        joinpath.(root, filter(contains(pattern), files))
    end

files = []
for pattern in [r"\.jld2", r"\.nc"]
    global files = vcat(files, recursive_find(@__DIR__, pattern))
end

for file in files
    rm(file)
end


# Replace with below once https://github.com/JuliaDocs/Documenter.jl/pull/2692 is merged and available.
#  deploydocs(repo = "github.com/FourierFlows/FourierFlows.jl",
#    deploy_repo = "github.com/FourierFlows/FourierFlowsDocumentation",
#    devbranch = "main",
#    forcepush = true,
#    push_preview = true,
#    versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"])

deploydocs(
    repo = "github.com/BiGSTARS/BiGSTARS.jl",
    deploy_repo = "github.com/BiGSTARS/BiGSTARSDocumentation",
    devbranch = "main",
    forcepush = true,
    push_preview = true,
    versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"]
)
