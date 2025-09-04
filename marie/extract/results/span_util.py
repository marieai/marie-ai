from typing import List, Optional

from marie.extract.models.base import Location
from marie.extract.models.exec_context import ExecutionContext
from marie.extract.models.match import MatchSection, ScanResult, Span
from marie.extract.models.page_span import PageSpan
from marie.extract.structures import UnstructuredDocument
from marie.extract.structures.line_with_meta import LineWithMeta


def pluck_lines_by_span(
    document: UnstructuredDocument, span: Span
) -> List[LineWithMeta]:
    """
    Given a document and a span, returns the lines from the document
    that fall within the span.
    """
    assert document is not None
    assert span is not None

    page_id = span.page
    lines = document.lines_for_page(page_id)
    start_line = span.start_line_id
    end_line = span.end_line_id

    return [line for line in lines if start_line <= line.metadata.line_id < end_line]


def create_page_span_from_lines(
    doc: UnstructuredDocument,
    start_line: Optional[LineWithMeta],
    end_line: Optional[LineWithMeta],
) -> PageSpan:
    """
    Creates a `PageSpan` based on the specified lines within a document, capturing the span of pages
    and vertical areas defined by the start and end lines. The resulting `PageSpan` combines all the
    intermediate page spans to represent the content between these lines inclusively.

    :param doc: The `UnstructuredDocument` instance containing the content and metadata of the document.
    :param start_line: Metadata of the starting line, including page and positional identifiers.
    :param end_line: Metadata of the ending line, including page and positional identifiers.
    :return: A `PageSpan` object that encapsulates the page and vertical ranges, defined by the lines.
    :raises ValueError: If the start page ID is greater than the end page ID, indicating an invalid range.
    """
    page_span = PageSpan()

    if not start_line or not end_line:
        return page_span

    start_page_id = start_line.metadata.page_id
    end_page_id = end_line.metadata.page_id

    if start_page_id > end_page_id:
        raise ValueError("Start page is after end page")

    for page_id in range(start_page_id, end_page_id + 1):
        lines_on_page = doc.lines_for_page(page_id)
        if not lines_on_page:
            continue

        all_line_ids = [line.metadata.line_id for line in lines_on_page]

        if start_page_id == end_page_id:
            y_start = start_line.metadata.line_id
            page_h = end_line.metadata.line_id - y_start + 1
            span = Span(page=page_id, y=y_start, h=max(0, page_h))
            page_span.add(span)
            break

        page_min_line_id = min(all_line_ids)
        page_max_line_id = max(all_line_ids)

        if page_id == start_page_id:
            y_start = start_line.metadata.line_id
            page_h = page_max_line_id - y_start + 1
        elif page_id == end_page_id:
            y_start = page_min_line_id
            page_h = end_line.metadata.line_id - y_start + 1
        else:
            y_start = page_min_line_id
            page_h = page_max_line_id - y_start + 1

        span = Span(page=page_id, y=y_start, h=max(0, page_h))
        page_span.add(span)

    return page_span


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


def pagespan_from_start_stop(
    context: ExecutionContext, start: ScanResult, stop: ScanResult, msg: str = ""
) -> PageSpan:
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
            if (
                sp == stop.page
            ):  # when the start and stop points are the same, the height of the span should be 1
                span.h = stop_y - start_y + 1
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
