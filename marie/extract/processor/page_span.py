from typing import List

from marie.extract.models.base import Location
from marie.extract.models.definition import ExecutionContext
from marie.extract.models.match import MatchSection, ScanResult
from marie.extract.models.span import Span


class PageSpan:

    def __init__(self):
        self.page_count: int = 0
        self.spanned_pages: List[Span] = []

    @staticmethod
    def from_context(
        context: ExecutionContext, owner: MatchSection, span: Span
    ) -> MatchSection:
        section = MatchSection()
        start = Location(page=span.page, y=span.y)
        stop = Location(page=span.page, y=span.y + span.h)

        section.y_offset = owner.y_offset
        section.start_candidates = owner.start_candidates
        section.stop_candidates = owner.stop_candidates
        section.owner_layer = owner.owner_layer

        section.pages = [context.get_page_by_page_number(span.page)]
        section.start = start
        section.stop = stop

        return section

    @staticmethod
    def create(
        context: ExecutionContext, start: ScanResult, stop: ScanResult, msg: str = ""
    ) -> 'PageSpan':
        assert context.document is not None
        doc = context.document
        print('Creating PAGE SPAN')
        print('Start:', start)
        print('Stop:', stop)

        page_span = PageSpan()
        sp = start.page
        ep = stop.page

        print(f"Start page = Page {sp}, Stop page = Page {ep}")
        # TODO: Implement two different strategies for start and stop (LINE and COORDINATE)

        for i in range(sp, ep + 1):
            lines_by_page = doc.lines_for_page(i)
            page_h = len(lines_by_page)
            start_y = start.line.metadata.line_id
            stop_y = stop.line.metadata.line_id

            span = Span(page=i, h=page_h, msg=msg)
            if i == sp:
                span.y = start_y
                if sp == stop.page:
                    span.h = stop_y - start_y
                else:
                    span.h = page_h - start_y
            elif i == ep:
                span.y = 0
                span.h = stop_y
            else:
                span.y = 0
            page_span.add(span)

        # for i in range(sp, ep + 1):
        #     span = Span(page=i, h=details.h, msg=msg)
        #
        #     if i == sp:
        #         span.y = start.y
        #         if sp == stop.page:
        #             span.h = stop.y - start.y
        #         else:
        #             span.h = details.h - start.y
        #     elif i == ep:
        #         span.y = 0
        #         span.h = stop.y
        #     else:
        #         span.y = 0
        #
        #     page_span.add(span)

        # Format and output the page spans
        print("Page spans:")
        for span in page_span.spanned_pages:
            print(
                f"Page: {span.page}, Start Y: {span.y}, Height: {span.h}, Message: {span.msg}"
            )

        return page_span

    def add(self, span: Span):
        if self.spanned_pages is None:
            self.spanned_pages = []

        self.page_count += 1
        self.spanned_pages.append(span)

    def get_spanned_page(self, page: int) -> Span:
        return self.get_spanned_page_from_list(self.spanned_pages, page)

    @staticmethod
    def get_spanned_page_from_list(spanned_pages: List[Span], page: int) -> Span:
        for span in spanned_pages:
            if span.page == page:
                return span
        raise ValueError("Span Page is out of bounds")

    @staticmethod
    def span_to_start_location(span: Span) -> Location:
        return Location(page=span.page, y=span.y)

    @staticmethod
    def span_to_stop_location(span: Span) -> Location:
        return Location(page=span.page, y=span.y + span.h)
