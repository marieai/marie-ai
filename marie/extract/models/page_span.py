from typing import List

from marie.extract.models.base import Location
from marie.extract.models.span import Span


class PageSpan:

    def __init__(self):
        self.page_count: int = 0
        self.spanned_pages: List[Span] = []

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
