#include "dialect_emitter.h"
#include "dialect_parser.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

    std::string read_text_file(const std::string &path)
    {
        std::ifstream stream(path, std::ios::binary);
        if (!stream)
        {
            throw std::runtime_error("failed to read file: " + path);
        }

        std::ostringstream buffer;
        buffer << stream.rdbuf();
        return buffer.str();
    }

    void write_text_file(const std::string &path, const std::string &text)
    {
        std::ofstream stream(path, std::ios::binary);
        if (!stream)
        {
            throw std::runtime_error("failed to write file: " + path);
        }
        stream << text;
    }

    bool starts_with(const std::string &value, const std::string &prefix)
    {
        return value.rfind(prefix, 0) == 0;
    }

    std::string quote_arg(const std::string &value)
    {
        if (value.empty())
        {
            return "\"\"";
        }

        bool requires_quotes = false;
        for (char ch : value)
        {
            if (ch == ' ' || ch == '\t' || ch == '"')
            {
                requires_quotes = true;
                break;
            }
        }

        if (!requires_quotes)
        {
            return value;
        }

        std::string out;
        out.reserve(value.size() + 2);
        out.push_back('"');
        for (char ch : value)
        {
            if (ch == '"')
            {
                out.push_back('\\');
            }
            out.push_back(ch);
        }
        out.push_back('"');
        return out;
    }

    std::string join_command(const std::vector<std::string> &argv)
    {
        std::string command;
        for (size_t index = 0; index < argv.size(); ++index)
        {
            if (index > 0)
            {
                command.push_back(' ');
            }
            command += quote_arg(argv[index]);
        }
        return command;
    }

    int run_command(const std::vector<std::string> &argv)
    {
        const std::string command = join_command(argv);
        const int rc = std::system(command.c_str());
        if (rc != 0)
        {
            std::cerr << "mini_triton_lower: command failed (" << rc << "): " << command << '\n';
        }
        return rc;
    }

    void print_usage()
    {
        std::cerr
            << "Usage: mini_triton_lower --mlir-opt <path> --mlir-translate <path> --llc <path> "
            << "[--input-dialect <path>] --input-mlir <path> --optimized-mlir <path> --llvm-ir <path> --ptx <path> "
            << "--cuda-arch <sm_xx>\n";
    }

} // namespace

int main(int argc, char **argv)
{
    std::unordered_map<std::string, std::string> args;
    for (int index = 1; index < argc; ++index)
    {
        std::string key = argv[index];
        if (!starts_with(key, "--"))
        {
            std::cerr << "mini_triton_lower: unexpected positional argument: " << key << '\n';
            print_usage();
            return 2;
        }

        if (index + 1 >= argc)
        {
            std::cerr << "mini_triton_lower: missing value for argument: " << key << '\n';
            print_usage();
            return 2;
        }

        args[key] = argv[++index];
    }

    const std::vector<std::string> required = {
        "--mlir-opt",
        "--mlir-translate",
        "--llc",
        "--input-mlir",
        "--optimized-mlir",
        "--llvm-ir",
        "--ptx",
        "--cuda-arch",
    };

    for (const std::string &key : required)
    {
        if (args.find(key) == args.end())
        {
            std::cerr << "mini_triton_lower: missing required argument: " << key << '\n';
            print_usage();
            return 2;
        }
    }

    try
    {
        auto dialect_it = args.find("--input-dialect");
        if (dialect_it != args.end())
        {
            const std::string dialect_text = read_text_file(dialect_it->second);
            const mtc_lower::DialectModule module = mtc_lower::parse_dialect_text(dialect_text);
            const std::string generated_mlir = mtc_lower::emit_mlir_from_dialect(module);
            write_text_file(args["--input-mlir"], generated_mlir);
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mini_triton_lower: failed to lower mini_triton dialect: " << ex.what() << '\n';
        return 2;
    }

    const int mlir_opt_rc = run_command({
        args["--mlir-opt"],
        args["--input-mlir"],
        "--convert-nvvm-to-llvm",
        "--reconcile-unrealized-casts",
        "-o",
        args["--optimized-mlir"],
    });
    if (mlir_opt_rc != 0)
    {
        return 1;
    }

    const int mlir_translate_rc = run_command({
        args["--mlir-translate"],
        "--mlir-to-llvmir",
        args["--optimized-mlir"],
        "-o",
        args["--llvm-ir"],
    });
    if (mlir_translate_rc != 0)
    {
        return 1;
    }

    const int llc_rc = run_command({
        args["--llc"],
        "-march=nvptx64",
        std::string("-mcpu=") + args["--cuda-arch"],
        args["--llvm-ir"],
        "-o",
        args["--ptx"],
    });
    if (llc_rc != 0)
    {
        return 1;
    }

    return 0;
}
