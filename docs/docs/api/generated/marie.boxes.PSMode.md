# marie.boxes.PSMode

### *class* marie.boxes.PSMode(value)

Page Segmentation Modes

* WORD = Treat the image as a single word.
* SPARSE = Sparse text. Find as much text as possible in no particular order.
* LINE = Treat the image as a single text line.
* RAW_LINE =  Raw line. Treat the image as a single text line, NO bounding box detection performed.
* MULTI_LINE = Multiline. Treat the image as multiple text lines, NO bounding box detection performed.

#### \_\_init_\_()

### Methods

| `from_value`(value)   | Returns the PSMode enum value corresponding to the given string value.   |
|-----------------------|--------------------------------------------------------------------------|

### Attributes

| `WORD`       |    |
|--------------|----|
| `SPARSE`     |    |
| `LINE`       |    |
| `RAW_LINE`   |    |
| `MULTI_LINE` |    |
