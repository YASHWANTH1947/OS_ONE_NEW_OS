#include "kernel.h"

typedef struct {
    uint16_t offset_low;
    uint16_t selector;
    uint8_t ist;
    uint8_t attributes;
    uint16_t offset_middle;
    uint32_t offset_high;
    uint32_t reserved;
} __attribute__((packed)) idt_entry_t;

typedef struct {
    uint16_t limit;
    uint64_t base;
} __attribute__((packed)) idt_pointer_t;

static idt_entry_t idt[256] __attribute__((aligned(16)));
extern const uintptr_t isr_stub_table[32];

static void set_gate(uint32_t vector, uintptr_t handler)
{
    idt[vector].offset_low = (uint16_t)handler;
    idt[vector].selector = 0x08u;
    idt[vector].ist = 0;
    idt[vector].attributes = 0x8Eu;
    idt[vector].offset_middle = (uint16_t)(handler >> 16);
    idt[vector].offset_high = (uint32_t)(handler >> 32);
    idt[vector].reserved = 0;
}

void idt_init(void)
{
    idt_pointer_t pointer;
    uint32_t vector;

    for (vector = 0; vector < 32u; ++vector) {
        set_gate(vector, isr_stub_table[vector]);
    }
    pointer.limit = (uint16_t)(sizeof(idt) - 1u);
    pointer.base = (uintptr_t)idt;
    __asm__ volatile ("lidt %0" : : "m"(pointer));
}

void exception_handler(uint64_t vector, uint64_t error, uint64_t rip)
{
    serial_printf("CPU exception %u: error=0x%llx RIP=0x%llx\n",
                  (uint32_t)vector, (unsigned long long)error,
                  (unsigned long long)rip);
    kernel_panic("unrecoverable CPU exception");
}
