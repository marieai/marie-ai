def main():
    from .parsers import get_main_parser

    args = get_main_parser().parse_args()

    try:
        from . import api

        getattr(api, args.auth_cli.replace('-', '_'))(args)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
