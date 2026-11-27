#!/usr/bin/env Rscript
# Use the same R workflow to render the exported vector files and check the PNG/PDF pair.
argv <- commandArgs(trailingOnly = TRUE)
stopifnot(length(argv) >= 1L)
out <- normalizePath(argv[1], winslash = "/", mustWork = TRUE)
figure_dir <- file.path(out, "figures")
qa_dir <- if (length(argv) >= 2) argv[2] else NULL
if (!is.null(qa_dir)) dir.create(qa_dir, recursive = TRUE, showWarnings = FALSE)

inspect <- function(path_pdf) {
  stem <- tools::file_path_sans_ext(basename(path_pdf))
  path_png <- file.path(figure_dir, paste0(stem, ".png"))
  info <- pdftools::pdf_info(path_pdf)
  size <- pdftools::pdf_pagesize(path_pdf)[1,]
  fonts <- pdftools::pdf_fonts(path_pdf)
  glyphs <- pdftools::pdf_data(path_pdf)[[1]]
  bitmap <- png::readPNG(path_png, info = TRUE)
  png_info <- attr(bitmap, "info")
  dim_png <- dim(bitmap)
  stopifnot(info$pages == 1L, nrow(glyphs) > 20L, nrow(fonts) >= 1L,
            all(fonts$embedded), all(glyphs$x >= 0), all(glyphs$y >= 0),
            all(glyphs$x + glyphs$width <= size$width + 1),
            all(glyphs$y + glyphs$height <= size$height + 1),
            all(png_info$dpi > 399), all(png_info$dpi < 401))
  pdf_raster <- pdftools::pdf_render_page(path_pdf, page = 1, dpi = 165)
  if (!is.null(qa_dir)) png::writePNG(pdf_raster, file.path(qa_dir, paste0(stem, ".png")))
  data.frame(figure = stem, raster_dpi = round(png_info$dpi[1]),
             width_mm = round(size$width*25.4/72), height_mm = round(size$height*25.4/72),
             width_px = dim_png[2], height_px = dim_png[1], pdf_pages = info$pages,
             pdf_fonts = paste(fonts$name, collapse = " | "), embedded_fonts = all(fonts$embedded),
             extractable_pdf_text = nrow(glyphs) > 20, pdf_renders = length(pdf_raster) > 0L)
}
paths <- list.files(figure_dir, pattern = "[.]pdf$", full.names = TRUE)
stopifnot(length(paths) == 7L)
checks <- do.call(rbind, lapply(paths, inspect))
utils::write.csv(checks, file.path(out, "FIGURE_EXPORT_QC.csv"), row.names = FALSE)
message("Verified ", nrow(checks), " PNG/PDF pairs.")
