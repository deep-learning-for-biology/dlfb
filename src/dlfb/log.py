import logging

logging.basicConfig(
  level=logging.INFO,
  format=(
    "%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(module)s - %(funcName)s: "
    "%(message)s"
  ),
  datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)

logging.getLogger("absl").addFilter(lambda _: False)
logging.getLogger("matplotlib").setLevel("WARNING")
