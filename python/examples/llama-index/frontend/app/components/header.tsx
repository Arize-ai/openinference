import Image from "next/image";

export default function Header() {
    return (
        <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
            <div className="fixed bottom-0 left-0 flex h-48 w-full items-end justify-center bg-gradient-to-t from-white via-white dark:from-black dark:via-black lg:static lg:h-auto lg:w-auto lg:bg-none">
                <a
                    href="https://www.llamaindex.ai/"
                    className="flex items-center justify-center font-nunito text-lg font-bold gap-2"
                >
                    <Image
                        className="rounded-xl"
                        src="/arize.png"
                        alt="Arize Logo"
                        width={40}
                        height={40}
                        priority
                    />
                    <span>Arize GPT</span>
                </a>
            </div>
        </div>
    );
}
