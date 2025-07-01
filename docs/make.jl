using Documenter, DocumenterCitations, Literate

using CairoMakie

using BiGSTARS

#####
##### Generate literated examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")

mkpath(OUTPUT_DIR)

examples = [
    "Stone1971.jl",
    "rRBC.jl"
]

@info "EXAMPLES_DIR: $EXAMPLES_DIR"
@info "OUTPUT_DIR: $OUTPUT_DIR"

# for example in examples
#     input_file = joinpath(EXAMPLES_DIR, example)
#     output_file = joinpath(OUTPUT_DIR, replace(example, ".jl" => ".md"))
#     Literate.markdown(input_file, output_file; 
#                       documenter=true, 
#                       include_comments=true, 
#                       include_code=true, 
#                       include_output=true)
# end

# for example in examples
#   withenv("GITHUB_REPOSITORY" => "github.com/BiGSTARS/BiGSTARSDocumentation") do
#     example_filepath = joinpath(EXAMPLES_DIR, example)
#     withenv("JULIA_DEBUG" => "Literate") do
#       Literate.markdown(example_filepath, OUTPUT_DIR;
#                         flavor = Literate.DocumenterFlavor(), execute = true)
#     end
#   end
# end

for example in examples
  try
    withenv("GITHUB_REPOSITORY" => "github.com/BiGSTARS/BiGSTARSDocumentation") do
      example_filepath = joinpath(EXAMPLES_DIR, example)
      withenv("JULIA_DEBUG" => "Literate") do
        Literate.markdown(example_filepath, OUTPUT_DIR;
                          flavor = Literate.DocumenterFlavor(), execute = true)
      end
    end
  catch e
    @error "Failed to process $example" exception=(e, catch_backtrace())
    rethrow()
  end
end

#####
##### Build and deploy docs
#####

# format = Documenter.HTML(
#     title = "BiGSTARS.jl",
#     authors = "BiGSTARS developers",
#     repo = ""
#     )


format = Documenter.HTML(
   collapselevel = 2,
      prettyurls = get(ENV, "CI", nothing) == "true",
  size_threshold = 2^21,
       canonical = "https://github.com/BiGSTARS/BiGSTARSDocumentation/stable/"
)


bib_filepath = joinpath(dirname(@__FILE__), "src/references.bib")
bib = CitationBibliography(bib_filepath, style=:authoryear)


#makedocs = Documenter.make_docs(

makedocs(
     authors = "Subhajit Kar, and contributors",
    sitename = "BiGSTARS.jl",
     modules = [BiGSTARS],
     plugins = [bib],
      format = format,
     doctest = true,
       clean = true,
   checkdocs = :all,
    pages = Any[
                "Home" => "index.md",
                "Installation" => "installation_instructions.md",
                "Examples" => [
                    "Ou1971" =>  Any[
                        "literated/Stone1971.md"
                        ],
                    "rRBC" => Any[
                        "literated/rRBC.md"
                        ]
                ],
                "Modules" => Any[
                    "modules/Ston1971.md",
                    "modules/rRBC.md",
                ],
                "Contributor's guide" => "contributing.md",
                "References" => "references.md"
    ]
)


# makedocs(
#      authors = "Subhajit Kar, and contributors",
#     sitename = "BiGSTARS.jl",
#      modules = [BiGSTARS],
#      plugins = [bib],
#       format = format,
#      doctest = true,
#        clean = true,
#    checkdocs = :all,
#     pages = Any[
#                 "Home" => "index.md",
#                 "Installation" => "installation_instructions.md",
#                 "Modules" => Any[
#                     "modules/Ou1971.md",
#                     "modules/rRBC.md",
#                 ],
#                 "Contributor's guide" => "contributing.md",
#                 "References" => "references.md"
#     ]
# )


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
# deploydocs(
#     repo = "github.com/BiGSTARS/BiGSTARS.jl",
#     deploy_repo = "github.com/BiGSTARS/BiGSTARSDocumentation",
#     devbranch = "main",
#     forcepush = true,
#     push_preview = true,
#     versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"]
# )


if get(ENV, "GITHUB_EVENT_NAME", "") == "pull_request"
    deploydocs(repo = "github.com/BiGSTARS/BiGSTARS.jl",
               repo_previews = "github.com/BiGSTARS/BiGSTARSDocumentation",
               devbranch = "main",
               forcepush = true,
               push_preview = true,
               versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"])
else
    repo = "github.com/BiGSTARS/BiGSTARSDocumentation"
    withenv("GITHUB_REPOSITORY" => repo) do
        deploydocs(; repo,
                     devbranch = "main",
                     forcepush = true,
                     versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"])
    end
end