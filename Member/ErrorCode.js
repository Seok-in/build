import swal from "sweetalert";

function ErrorCode(result) {
  const code = result.code;

  if (code === 596) {
    swal(`${result.message}`);
    return;
  }

  switch (code) {
    case 300:
      swal("허용되지 않습니다");
      break;
    case 500:
      swal("내부 서버 에러");
      break;
    case 501:
      swal("로그인을 해주세요");
      break;
    case 502:
      swal("로그인 실패");
      break;
    case 503:
      swal("멤버가 아닙니다");
      break;
    case 504:
      swal("없는 멤버입니다");
      break;
    case 505:
      swal("이메일 키가 옳지 않습니다");
      break;
    case 506:
      swal("이메일 키가 만료되었습니다");
      break;
    case 509:
      swal("이미 회원가입 되어 있습니다");
      break;
    case 510:
      swal("해당 그룹이 없습니다");
      break;
    case 511:
      swal("카테고리가 없습니다");
      break;
    case 512:
      swal("파일 업로드에 실패했습니다");
      break;
    case 513:
      swal("패스워드가 틀렸습니다");
      break;
    case 514:
      swal("중복된 이메일입니다");
      break;
    case 515:
      swal("중복된 닉네임입니다");
      break;
    case 516:
      swal("비밀번호 길이가 맞지 않습니다");
      break;
    case 520:
      swal("관련 정보가 없습니다");
      break;
    case 521:
      swal("허용되지 않는 값입니다");
      break;
    case 522:
      swal("없는 방입니다");
      break;
    case 523:
      swal("참가자가 아닙니다");
      break;
    case 524:
      swal("같은 이름이 존재합니다");
      break;
    case 525:
      swal("비밀번호가 숫자가 아닙니다");
      break;
    case 526:
      swal("허용되지 않는 인원수입니다");
      break;
    case 527:
      swal("허용되지 않는 종료기간입니다");
      break;
    case 528:
      swal("이미지 파일이 아닙니다");
      break;
    case 529:
      swal("이미지 파일의 크기는 10MB 이하여야 합니다");
      break;
    case 530:
      swal("방장이 아닙니다");
      break;
    case 531:
      swal("스터디 로그 생성을 실패했습니다");
      break;
    case 532:
      swal("스터디 로그 저장을 실패했습니다");
      break;
    case 533:
      swal("스프린트가 없습니다");
      break;
    case 534:
      swal("이미 참가되었습니다");
      break;
    case 535:
      swal("내가 참여한 그룹이 아닙니다");
      break;
    case 536:
      swal("방장이 아닙니다");
      break;
    case 537:
      swal("모집중인 스프린트가 아닙니다");
      break;
    case 538:
      swal("포인트가 부족합니다");
      break;
    case 540:
      swal("참가 신청하지 않았습니다");
      break;
    case 541:
      swal("진행중인 스프린트가 아닙니다");
      break;
    case 543:
      swal("스프린트가 이미 시작되었습니다");
      break;
    case 544:
      swal("허용되는 시작/종료일자가 아닙니다");
      break;
    case 550:
      swal("인원 초과입니다");
      break;
    case 551:
      swal("비밀번호는 숫자만 가능합니다");
      break;
    case 552:
      swal("참가신청이 반려되었습니다");
      break;
    case 564:
      swal("비밀번호는 숫자만 가능합니다");
      break;
    case 565:
      swal("허용되지 않습니다");
      break;
    case 566:
      swal("반려되었습니다");
      break;
    case 567:
      swal("이미 신청되었습니다");
      break;
    case 595:
      swal("비정상적인 접근입니다");
      break;
    case 597:
      swal("카카오페이 오류입니다");
      break;
    case 598:
      swal("허용되지 않는 포인트 값입니다");
      break;
    case 599:
      swal("날짜 파싱을 실패했습니다");
      break;
  }
}

export default ErrorCode;
