import aiohttp
import asyncio

MAX_CONCURRENT = 2
sem = asyncio.Semaphore(MAX_CONCURRENT)

async def get_content(url, save_path):
    print('Requesting {}'.format(url))
    async with sem:
        async with aiohttp.ClientSession() as session:
            print('Session created')
            async with session.get(url) as response:
                print('Response {} received'.format(response.status))
                if response.status == 200:
                    data = await response.read()
                    print('Content extracted')
                    with open(save_path, 'wb') as f:
                        f.write(data)


async def main():
    urls = [f'http://os.cs.tsinghua.edu.cn/oscourse/OS2020spring/lecture21?action=AttachFile&do=get&target=slide-21-0{i}.pdf' for i in range(1, 6)]
    save_paths = [f'slide-21-0{i}.pdf' for i in range(1, 6)]

    tasks = [get_content(url, save_path) for url, save_path in zip(urls, save_paths)]
    await asyncio.gather(*tasks)
    print('All done')


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
