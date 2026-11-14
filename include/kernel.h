#ifndef KERNEL_H
#define KERNEL_H

#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------------------- */
/* Port I/O                                                                   */
/* ------------------------------------------------------------------------- */

static inline void port_out(uint16_t port, uint8_t value)
{
    __asm__ volatile ("outb %0, %1" : : "a"(value), "Nd"(port));
}

static inline uint8_t port_in(uint16_t port)
{
    uint8_t value;

    __asm__ volatile ("inb %1, %0" : "=a"(value) : "Nd"(port));
    return value;
}

/* ------------------------------------------------------------------------- */
/* Serial console                                                             */
/* ------------------------------------------------------------------------- */

void serial_init(void);
void serial_write(const char *text);
int serial_available(void);
char serial_read(void);
void serial_printf(const char *format, ...);
void serial_repeat(char character, uint32_t count);
void serial_rule(void);
void serial_aligned(const char *label, const char *value);
size_t serial_length(const char *text);
void kernel_panic(const char *message);

/* ------------------------------------------------------------------------- */
/* Physical and virtual memory                                               */
/* ------------------------------------------------------------------------- */

int pmm_init(uint32_t multiboot_info);
uint64_t pmm_alloc_frame(void);
int pmm_free_frame(uint64_t address);
uint32_t pmm_free_frames(void);
uint64_t pmm_usable_bytes(void);
void *kmalloc(size_t size);
size_t kmalloc_capacity(void);
size_t kmalloc_used(void);
int vmm_map_2m(uint64_t virtual_address, uint64_t physical_address,
               uint64_t flags);

/* ------------------------------------------------------------------------- */
/* Interrupts, timing, machine control                                       */
/* ------------------------------------------------------------------------- */

void idt_init(void);
void exception_handler(uint64_t vector, uint64_t error, uint64_t rip);
uint64_t read_tsc(void);
uint64_t tsc_calibrate(void);
void qemu_debug_exit(uint32_t code);

extern char _kernel_start[];
extern char _kernel_end[];

#endif
