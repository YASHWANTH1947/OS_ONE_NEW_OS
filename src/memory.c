#include "kernel.h"

#define PAGE_SIZE 4096u
#define MAX_PHYSICAL_ADDRESS (1ULL << 30)
#define MAX_FRAMES (MAX_PHYSICAL_ADDRESS / PAGE_SIZE)
#define BITMAP_BYTES (MAX_FRAMES / 8u)
#define MULTIBOOT_TAG_END 0u
#define MULTIBOOT_TAG_MODULE 3u
#define MULTIBOOT_TAG_MMAP 6u
#define MULTIBOOT_MEMORY_AVAILABLE 1u

typedef struct {
    uint32_t total_size;
    uint32_t reserved;
} multiboot_info_t;

typedef struct {
    uint32_t type;
    uint32_t size;
} multiboot_tag_t;

typedef struct {
    uint32_t type;
    uint32_t size;
    uint32_t start;
    uint32_t end;
    char command_line[];
} multiboot_module_t;

typedef struct {
    uint64_t address;
    uint64_t length;
    uint32_t type;
    uint32_t reserved;
} multiboot_mmap_entry_t;

typedef struct {
    uint32_t type;
    uint32_t size;
    uint32_t entry_size;
    uint32_t entry_version;
    uint8_t entries[];
} multiboot_mmap_tag_t;

static uint8_t frame_bitmap[BITMAP_BYTES];
static uint8_t allocated_bitmap[BITMAP_BYTES];
static uint32_t free_frames;
static uint64_t usable_bytes;
static uint8_t kernel_heap[65536] __attribute__((aligned(16)));
static size_t kernel_heap_used;

extern uint64_t page_directory[512];

static void mark_frame(uint64_t frame, int used)
{
    uint8_t mask = (uint8_t)(1u << (frame & 7u));

    if (used) {
        frame_bitmap[frame >> 3] |= mask;
    } else {
        frame_bitmap[frame >> 3] &= (uint8_t)~mask;
    }
}

static int frame_is_used(uint64_t frame)
{
    return (frame_bitmap[frame >> 3] & (uint8_t)(1u << (frame & 7u))) != 0;
}

static void mark_allocated(uint64_t frame, int allocated)
{
    uint8_t mask = (uint8_t)(1u << (frame & 7u));

    if (allocated) {
        allocated_bitmap[frame >> 3] |= mask;
    } else {
        allocated_bitmap[frame >> 3] &= (uint8_t)~mask;
    }
}

static int frame_is_allocated(uint64_t frame)
{
    return (allocated_bitmap[frame >> 3] &
            (uint8_t)(1u << (frame & 7u))) != 0;
}

static void reserve_range(uint64_t start, uint64_t end)
{
    uint64_t first;
    uint64_t last;
    uint64_t frame;

    if (end <= start) {
        return;
    }
    first = start / PAGE_SIZE;
    last = end / PAGE_SIZE + (end % PAGE_SIZE != 0);
    if (last > MAX_FRAMES) {
        last = MAX_FRAMES;
    }
    for (frame = first; frame < last; ++frame) {
        if (!frame_is_used(frame)) {
            mark_frame(frame, 1);
            --free_frames;
        }
    }
}

int pmm_init(uint32_t multiboot_info)
{
    const multiboot_info_t *info =
        (const multiboot_info_t *)(uintptr_t)multiboot_info;
    uint32_t offset = sizeof(*info);
    int found_map = 0;
    uint32_t i;

    for (i = 0; i < BITMAP_BYTES; ++i) {
        frame_bitmap[i] = 0xFFu;
        allocated_bitmap[i] = 0;
    }
    free_frames = 0;
    usable_bytes = 0;

    while (offset + sizeof(multiboot_tag_t) <= info->total_size) {
        const multiboot_tag_t *tag =
            (const multiboot_tag_t *)((const uint8_t *)info + offset);

        if (tag->size < sizeof(*tag) || tag->size > info->total_size - offset) {
            return 0;
        }
        if (tag->type == MULTIBOOT_TAG_END) {
            break;
        }
        if (tag->type == MULTIBOOT_TAG_MMAP &&
            tag->size >= sizeof(multiboot_mmap_tag_t)) {
            const multiboot_mmap_tag_t *map =
                (const multiboot_mmap_tag_t *)tag;
            uint32_t entry_offset;

            if (map->entry_size < sizeof(multiboot_mmap_entry_t)) {
                return 0;
            }
            if (tag->size - sizeof(multiboot_mmap_tag_t) < map->entry_size) {
                return 0;
            }
            found_map = 1;
            for (entry_offset = 0;
                 entry_offset <= tag->size - sizeof(multiboot_mmap_tag_t) &&
                     map->entry_size <= tag->size -
                         sizeof(multiboot_mmap_tag_t) - entry_offset;
                 entry_offset += map->entry_size) {
                const multiboot_mmap_entry_t *entry =
                    (const multiboot_mmap_entry_t *)(map->entries +
                                                     entry_offset);
                uint64_t end;
                uint64_t first;
                uint64_t last;
                uint64_t frame;

                if (entry->type != MULTIBOOT_MEMORY_AVAILABLE ||
                    entry->length > UINT64_MAX - entry->address) {
                    continue;
                }
                end = entry->address + entry->length;
                if (end <= PAGE_SIZE || entry->address >= MAX_PHYSICAL_ADDRESS) {
                    continue;
                }
                first = (entry->address + PAGE_SIZE - 1u) / PAGE_SIZE;
                if (first < 0x100000u / PAGE_SIZE) {
                    first = 0x100000u / PAGE_SIZE;
                }
                last = end / PAGE_SIZE;
                if (last > MAX_FRAMES) {
                    last = MAX_FRAMES;
                }
                if (last > first) {
                    usable_bytes += (last - first) * PAGE_SIZE;
                }
                for (frame = first; frame < last; ++frame) {
                    if (frame_is_used(frame)) {
                        mark_frame(frame, 0);
                        ++free_frames;
                    }
                }
            }
        }
        offset = (offset + tag->size + 7u) & ~7u;
    }

    if (!found_map) {
        return 0;
    }
    reserve_range(0, 0x100000u);
    reserve_range((uint64_t)(uintptr_t)_kernel_start,
                  (uint64_t)(uintptr_t)_kernel_end);
    reserve_range(multiboot_info,
                  (uint64_t)multiboot_info + info->total_size);

    offset = sizeof(*info);
    while (offset + sizeof(multiboot_tag_t) <= info->total_size) {
        const multiboot_tag_t *tag =
            (const multiboot_tag_t *)((const uint8_t *)info + offset);

        if (tag->size < sizeof(*tag) || tag->size > info->total_size - offset) {
            return 0;
        }
        if (tag->type == MULTIBOOT_TAG_END) {
            break;
        }
        if (tag->type == MULTIBOOT_TAG_MODULE &&
            tag->size >= sizeof(multiboot_module_t)) {
            const multiboot_module_t *module =
                (const multiboot_module_t *)tag;
            reserve_range(module->start, module->end);
        }
        offset = (offset + tag->size + 7u) & ~7u;
    }
    return 1;
}

uint64_t pmm_alloc_frame(void)
{
    uint64_t frame;

    for (frame = PAGE_SIZE / PAGE_SIZE; frame < MAX_FRAMES; ++frame) {
        if (!frame_is_used(frame)) {
            mark_frame(frame, 1);
            mark_allocated(frame, 1);
            --free_frames;
            return frame * PAGE_SIZE;
        }
    }
    return 0;
}

int pmm_free_frame(uint64_t address)
{
    uint64_t frame;

    if ((address & (PAGE_SIZE - 1u)) != 0 ||
        address >= MAX_PHYSICAL_ADDRESS) {
        return 0;
    }
    frame = address / PAGE_SIZE;
    if (!frame_is_allocated(frame)) {
        return 0;
    }
    mark_frame(frame, 0);
    mark_allocated(frame, 0);
    ++free_frames;
    return 1;
}

uint32_t pmm_free_frames(void)
{
    return free_frames;
}

uint64_t pmm_usable_bytes(void)
{
    return usable_bytes;
}

size_t kmalloc_capacity(void)
{
    return sizeof(kernel_heap);
}

size_t kmalloc_used(void)
{
    return kernel_heap_used;
}

void *kmalloc(size_t size)
{
    size_t aligned = (size + 15u) & ~(size_t)15u;
    void *result;

    if (aligned < size || aligned > sizeof(kernel_heap) - kernel_heap_used) {
        return 0;
    }
    result = &kernel_heap[kernel_heap_used];
    kernel_heap_used += aligned;
    return result;
}

int vmm_map_2m(uint64_t virtual_address, uint64_t physical_address,
               uint64_t flags)
{
    uint64_t index;
    uint64_t entry;

    if ((virtual_address & 0x1FFFFFu) != 0 ||
        (physical_address & 0x1FFFFFu) != 0 ||
        virtual_address >= MAX_PHYSICAL_ADDRESS ||
        physical_address >= MAX_PHYSICAL_ADDRESS) {
        return 0;
    }
    index = virtual_address / 0x200000u;
    entry = (physical_address & 0x000FFFFFFFE00000ULL) |
            (flags & 0xFFFu) | 0x83u;
    page_directory[index] = entry;
    __asm__ volatile ("invlpg (%0)" : : "r"(virtual_address) : "memory");
    return 1;
}
