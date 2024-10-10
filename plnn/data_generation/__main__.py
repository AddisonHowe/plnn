import sys

def main():
    from plnn.data_generation.generate_data import parse_args, main
    args = parse_args(sys.argv[1:])
    main(args)

if __name__ == "__main__":
    main()
