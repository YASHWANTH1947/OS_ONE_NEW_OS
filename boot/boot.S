.section .multiboot2,"a"
.align 8
multiboot_header:
    .long 0xE85250D6
    .long 0
    .long multiboot_header_end - multiboot_header
    .long -(0xE85250D6 + (multiboot_header_end - multiboot_header))
    .short 0
    .short 0
    .long 8
    .short 5
    .short 0
    .long 20
    .long 1024
    .long 768
    .long 32
multiboot_header_end:

.section .text.boot,"ax"
.code32
.global _start
.type _start, @function
_start:
    cli
    cld
    cmpl $0x36D76289, %eax
    jne boot_halt

    movl %ebx, boot_mbi_ptr
    movl $boot_stack_top, %esp
    andl $-16, %esp

    movl $page_directory, %edi
    movl $0x83, %eax
    movl $512, %ecx
1:
    movl %eax, (%edi)
    movl $0, 4(%edi)
    addl $0x200000, %eax
    addl $8, %edi
    loop 1b

    movl $page_directory, %eax
    orl $0x3, %eax
    movl %eax, pdpt
    movl $pdpt, %eax
    orl $0x3, %eax
    movl %eax, pml4

    movl %cr4, %eax
    orl $0x620, %eax
    movl %eax, %cr4

    movl $0xC0000080, %ecx
    rdmsr
    orl $0x100, %eax
    wrmsr

    movl $pml4, %eax
    movl %eax, %cr3

    movl %cr0, %eax
    andl $~0xC, %eax
    orl $0x80000002, %eax
    movl %eax, %cr0

    lgdt gdt64_pointer
    ljmp $0x08, $long_mode_start

boot_halt:
    cli
2:
    hlt
    jmp 2b

.code64
long_mode_start:
    movw $0x10, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %ss
    xorw %ax, %ax
    movw %ax, %fs
    movw %ax, %gs

    movq $boot_stack_top, %rsp
    andq $-16, %rsp
    movl boot_mbi_ptr(%rip), %edi
    fninit
    ldmxcsr initial_mxcsr(%rip)
    call kernel_main

    cli
3:
    hlt
    jmp 3b

.align 8
gdt64:
    .quad 0
    .quad 0x00AF9A000000FFFF
    .quad 0x00CF92000000FFFF
gdt64_end:
gdt64_pointer:
    .short gdt64_end - gdt64 - 1
    .long gdt64

.align 4
initial_mxcsr:
    .long 0x1F80
boot_mbi_ptr:
    .long 0

.section .bss,"aw",@nobits
.align 4096
.global pml4
pml4:
    .skip 4096
.global pdpt
pdpt:
    .skip 4096
.global page_directory
page_directory:
    .skip 4096

.align 16
boot_stack_bottom:
    .skip 16384
boot_stack_top:

.section .note.GNU-stack,"",@progbits
