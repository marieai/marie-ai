def main():
    from .parsers import get_main_parser

    args = get_main_parser().parse_args()

    try:
        from .hubio import HubIO

        getattr(HubIO(args), args.hub_cli.replace('-', '_'))()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
