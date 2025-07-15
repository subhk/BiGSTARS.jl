using Documenter
using DocumenterCitations
using Literate

using CairoMakie
using Printf

using BiGSTARS

#####
##### Generate literated examples
#####

# using DocumenterTools
# DocumenterTools.generate_keypair(
#   repo="subhk/BiGSTARSDocumentation",
#   dir="."
# )

bib_filepath = joinpath(dirname(@__FILE__), "src/references.bib")
bib = CitationBibliography(bib_filepath, style=:authoryear)

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")

examples = [
    "Stone1971.jl",
    # "Eady.jl",
    "rRBC.jl"
]

# for example in examples
#     input_file = joinpath(EXAMPLES_DIR, example)
#     output_file = joinpath(OUTPUT_DIR, "") #replace(example, ".jl" => ".md"))
#     Literate.markdown(input_file, 
#                     output_file; 
#                     documenter=true, 
#                     include_comments=true, 
#                     include_code=true, 
#                     include_output=true)
# end

for example in examples
  withenv("GITHUB_REPOSITORY" => "subhk/BiGSTARSDocumentation") do
    example_filepath = joinpath(EXAMPLES_DIR, example)
    withenv("JULIA_DEBUG" => "Literate") do
      Literate.markdown(example_filepath, 
                        OUTPUT_DIR;
                        flavor = Literate.DocumenterFlavor(), 
                        execute = true)
    end
  end
end

# for example in examples
#   example_filepath = joinpath(EXAMPLES_DIR, example)
#   withenv("GITHUB_REPOSITORY" => "subhk/BiGSTARSDocumentation") do
#     Literate.markdown(example_filepath, OUTPUT_DIR; flavor = Literate.DocumenterFlavor())
#     Literate.notebook(example_filepath, OUTPUT_DIR)
#     Literate.script(example_filepath, OUTPUT_DIR)
#   end
# end


#####
##### Build and deploy docs
#####

format = Documenter.HTML(
    collapselevel  = 2,
    prettyurls     = get(ENV, "CI", nothing) == "true",
    size_threshold      = 250 * 1024^2,   # 100 MiB
    size_threshold_warn =  20 * 1024^2,    #  20 MiB warning
    canonical      = "https://subhk.github.io/BiGSTARSDocumentation/stable"
)


# pages = [
#     "Home" => "index.md",
#     "Installation Instructions" => "installation_instructions.md",
#     "Differentiation matrix"    => "matrices.md",
#     "Examples"                  => [ 
#       "Stone(1971) API"         => Any[
#                 "literated/Stone1971.md",
#       ],
#     ],
#     "Contributor's guide" => "contributing.md",
#     "References" => "references.md",
# ]

@printf("Building doc ...\n")

# makedocs(sitename = "BiGSTARS.jl",
#           authors = "Subhajit Kar, and contributors",
#           modules = [BiGSTARS],
#            format = format,
#             pages = pages,
#           plugins = [bib],
#           doctest = true,
#          warnonly = [:cross_references],
#             clean = true,
#         checkdocs = :exports)


makedocs(
    format    = format,
    authors   = "Subhajit Kar and contributors",
    sitename  = "BiGSTARS.jl",
    modules   = [BiGSTARS],
    plugins   = [bib],
    doctest   = true,
    clean     = true,
    checkdocs = :warn,
    warnonly   = [:cross_references],
    pages     = Any[
                "Home"                          => "index.md",
                "Installation instructions"     => "installation_instructions.md",
                "Differentiation matrix"        => "matrices.md",
                "Methodology"                   => "method.md",
                "Examples"                      => Any[
                    "Stone1971"                 => "literated/Stone1971.md", 
                    # "Eady"                      => "literated/Eady.md",
                    "rRBC"                      => "literated/rRBC.md"
                ],
                # "Modules"                   => Any[
                #     "Stone1971 API"         => "modules/Stone1971.md",
                #     # "rRBC API"              => "modules/rRBC.md"
                # ],
                # "Examples" => [ 
                #     "literated/Stone1971.md",
                # ],
                "Contributor's Guide"       => "contributing.md",
                "References"                => "references.md"
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
# deploydocs(
#     repo = "github.com/BiGSTARS/BiGSTARS.jl",
#     deploy_repo = "github.com/BiGSTARS/BiGSTARSDocumentation",
#     devbranch = "main",
#     forcepush = true,
#     push_preview = true,
#     versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"]
# )


if get(ENV, "GITHUB_EVENT_NAME", "") == "pull_request"
    deploydocs(repo = "github.com/subhk/BiGSTARS.jl",
               repo_previews = "github.com/subhk/BiGSTARSDocumentation",
               devbranch = "main",
               forcepush = true,
               push_preview = true,
               versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"])
else
    repo = "github.com/subhk/BiGSTARSDocumentation"
    withenv("GITHUB_REPOSITORY" => repo) do
        deploydocs(; repo,
                     devbranch = "main",
                     forcepush = true,
                     versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"])
    end
end