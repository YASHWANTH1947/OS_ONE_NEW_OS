ENTRY(_start)

SECTIONS
{
    . = 1M;
    _kernel_start = .;

    .multiboot2 ALIGN(8) :
    {
        KEEP(*(.multiboot2))
    }

    .text ALIGN(4K) :
    {
        *(.text.boot)
        *(.text .text.*)
    }

    .rodata ALIGN(4K) :
    {
        *(.rodata .rodata.*)
    }

    .data ALIGN(4K) :
    {
        *(.data .data.*)
    }

    .bss ALIGN(4K) (NOLOAD) :
    {
        *(COMMON)
        *(.bss .bss.*)
    }

    _kernel_end = .;
    /DISCARD/ :
    {
        *(.eh_frame)
        *(.note .note.*)
        *(.comment)
    }
}
