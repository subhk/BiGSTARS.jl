using Documenter 
using DocumenterCitations
using Literate

using Documenter: Remotes

using BiGSTARS

using StaticArrays
using SpecialFunctions
using CairoMakie

using Literate: DocumenterFlavor

#####
##### Generate literated examples
#####

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
bib          = CitationBibliography(bib_filepath, style = :authoryear)

@printf "making makedocs... \n"

try
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
            # "Modules"             => [
            #     "Stone1971 API"   => "src/Stone1971.md",
            #     "rRBC API"        => "rRBC.md"
            # ],
            "Contributor's Guide" => "contributing.md",
            "References"          => "references.md"
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

files = String[]
for pattern in [r"\\.jld2", r"\\.nc"]
    append!(files, recursive_find(@__DIR__, pattern))
end

for file in files
    rm(file; force = true)
end


# if get(ENV, "GITHUB_EVENT_NAME", "") == "pull_request"
#     deploydocs(repo = "subhk/BiGSTARS.jl",
#                repo_previews = "subhk/BiGSTARSDocumentation.git",
#                devbranch = "main",
#                forcepush = true,
#                push_preview = true,
#                versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"])
# else
#     repo = "subhk/BiGSTARSDocumentation.git"
#     withenv("GITHUB_REPOSITORY" => repo) do
#         deploydocs(; repo,
#                      devbranch = "main",
#                      forcepush = true,
#                      versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"])
#     end
# end


# if get(ENV, "GITHUB_EVENT_NAME", "") == "pull_request"
#     deploydocs(
#         repo = "git@github.com:subhk/BiGSTARSDocumentation.git",
#         devbranch = "main",
#         forcepush = true,
#         push_preview = true,
#         versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"]
#     )
# else
#     repo = "git@github.com:subhk/BiGSTARSDocumentation.git"
#     withenv("GITHUB_REPOSITORY" => repo) do
#         deploydocs(; repo,
#                      devbranch = "main",
#                      forcepush = true,
#                      versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"])
#     end
# end



# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
if haskey(ENV, "GITHUB_REPOSITORY")  # if we're on github
    deploydocs(;
        repo = "github.com/subhk/BiGSTARS.jl.git",
        branch = "gh-pages",
        devbranch = "master",
        forcepush = true,
        push_preview = true,
    )
end

# deploydocs setup remains as-is...