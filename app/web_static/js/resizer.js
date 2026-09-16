// 拖动分隔条改变区块尺寸。四个手柄共用（左栏、右栏、工具栏、输入区）。
//
// 这四处的交互完全一样：pointer 捕获、双击复位、方向键微调、松手后持久化。
// 复制四份迟早会分叉——尤其是「方向键的符号」这种细节，右栏往右按是变小、左栏
// 往右按是变大，写错一处不会报错，只是手感不对。

const STORAGE_KEY = "rag-rail-widths";
const STEP = 12;
const BIG_STEP = 40;

/** 单栏最多占视口这么大比例：在 1400px 窗口设的宽度，换个 1000px 的窗口会溢出。 */
const MAX_VIEWPORT_RATIO = 0.6;

function clampSize(value, limits, axis) {
  let max = limits.max;
  if (axis === "x") {
    max = Math.min(max, Math.max(limits.min, Math.round(window.innerWidth * MAX_VIEWPORT_RATIO)));
  }
  return Math.round(Math.min(max, Math.max(limits.min, value)));
}

/** 读回所有存过的尺寸。隐私模式或数据畸形都返回空对象。 */
export function readLayoutSizes() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return saved && typeof saved === "object" ? saved : {};
  } catch (_) {
    return {};
  }
}

export function writeLayoutSize(field, value) {
  const sizes = readLayoutSizes();
  sizes[field] = Math.round(value);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sizes));
  } catch (_) {
    /* localStorage 不可用（隐私模式）：尺寸只在本次会话生效 */
  }
}

/**
 * 给一个手柄装上拖拽。
 *
 * - `axis`: "x" 改宽度、"y" 改高度
 * - `growsWithDrag`: 鼠标往正方向移动时尺寸变大还是变小（+1 / -1）
 * - `read` / `apply`: 怎么量当前尺寸、怎么把尺寸写回去（CSS 变量）
 * - `field`: 存在 localStorage 里的字段名
 *
 * 启动时会自动恢复存过的值；窄屏下手柄可能被 CSS 隐藏，那时值仍留着不生效。
 */
export function enableResize({ handle, axis, growsWithDrag, limits, read, apply, field }) {
  if (!handle) return;

  const set = (size) => {
    const clamped = clampSize(size, limits, axis);
    apply(clamped);
    return clamped;
  };
  const current = () => read() || limits.default;

  let start = 0;
  let startSize = 0;
  let dragging = false;

  const persist = () => writeLayoutSize(field, current());

  const saved = readLayoutSizes()[field];
  if (Number.isFinite(saved)) set(saved);

  handle.addEventListener("pointerdown", (event) => {
    dragging = true;
    start = axis === "x" ? event.clientX : event.clientY;
    startSize = current();
    handle.classList.add("dragging");
    handle.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  handle.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const moved = (axis === "x" ? event.clientX : event.clientY) - start;
    set(startSize + growsWithDrag * moved);
  });

  const stop = (event) => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("dragging");
    if (handle.hasPointerCapture(event.pointerId)) {
      handle.releasePointerCapture(event.pointerId);
    }
    persist();
  };
  handle.addEventListener("pointerup", stop);
  handle.addEventListener("pointercancel", stop);

  handle.addEventListener("dblclick", () => {
    set(limits.default);
    persist();
  });

  handle.addEventListener("keydown", (event) => {
    const positive = axis === "x" ? "ArrowRight" : "ArrowDown";
    const negative = axis === "x" ? "ArrowLeft" : "ArrowUp";
    // 按「正方向」的箭头时尺寸朝哪边走，由 growsWithDrag 决定：右栏在右边，
    // 往右按是变窄；左栏相反。
    let direction = 0;
    if (event.key === positive) direction = growsWithDrag;
    else if (event.key === negative) direction = -growsWithDrag;
    else return;
    event.preventDefault();
    set(current() + direction * (event.shiftKey ? BIG_STEP : STEP));
    persist();
  });
}
